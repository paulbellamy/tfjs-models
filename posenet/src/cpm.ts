/**
 * @license
 * Copyright 2019 Google Inc. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * https://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */

import * as tfconv from '@tensorflow/tfjs-converter';
import * as tf from '@tensorflow/tfjs-core';

import { BaseModel } from './posenet_model';
import { argmax2d } from './single_pose/argmax2d';
import {
  Pose,
  PosenetInput,
  SinglePersonInterfaceConfig,
  PoseNetInputResolution,
  PoseNetOutputStride
} from './types';
import { getInputTensorDimensions, padAndResizeTo } from './util';

function toFloatIfInt(input: tf.Tensor3D): tf.Tensor3D {
  return tf.tidy(() => {
    if (input.dtype === 'int32') {
      input = input.toFloat();
    }
    const imageNetMean = tf.tensor([-123.15, -115.9, -103.06]);
    return input.add(imageNetMean);
  });
}

const imageSizeX = 192;
const imageSizeY = 192;
const outputW = 96;
const outputH = 96;
// const numBytesPerChannel = 4;
//
const bodyDict = [
  'nose',
  'neck',
  'leftShoulder',
  'leftElbow',
  'leftWrist',
  'rightShoulder',
  'rightElbow',
  'rightWrist',
  'leftHip',
  'leftKnee',
  'leftAnkle',
  'rightHip',
  'rightKnee',
  'rightAnkle'
];

export class CPM implements BaseModel {
  readonly model: tfconv.GraphModel;
  readonly outputStride: PoseNetOutputStride;

  constructor(model: tfconv.GraphModel, outputStride: PoseNetOutputStride) {
    this.model = model;
    /*
    const inputShape = this.model.inputs[0].shape as [
      number,
      number,
      number,
      number
    ];
    tf.util.assert(
      inputShape[1] === -1 && inputShape[2] === -1,
      () =>
        `Input shape [${inputShape[1]}, ${inputShape[2]}] ` +
        `must both be equal to or -1`
    );
    */
    this.outputStride = outputStride;
  }

  async estimateSinglePose(
    input: PosenetInput,
    inputResolution: PoseNetInputResolution,
    config: SinglePersonInterfaceConfig
  ): Promise<Pose> {
    const { resized } = padAndResizeTo(input, [imageSizeX, imageSizeY]);
    const heatmap = this.predict(resized).heatmapScores;
    console.log('heatmap:', await heatmap.data());
    const maxIdxs = argmax2d(heatmap); // 14 x 2
    console.log('maxIdxs shape:', maxIdxs.shape);
    const idxs = await maxIdxs.data();
    console.log('maxIdxs:', idxs);
    const [inputHeight, inputWidth] = getInputTensorDimensions(input);
    const keypoints = bodyDict.map((part, i) => {
      const yOrig = idxs[i + 14];
      const y = (yOrig * inputHeight) / outputH;
      const xOrig = idxs[i];
      const x = (xOrig * inputWidth) / outputW;
      console.log('(', xOrig, ',', yOrig, ') -> (', x, ', ', y, ')');
      return {
        part,
        position: { y, x },
        score: 1 // TODO: Figure out how to get the scale as well. Should be from the heatmap
      };
    });

    // const outputs = this.model.outputs[0].shape as [
    //   number,
    //   number,
    //   number,
    //   number
    // ];
    // const mpDim = outputs[1];
    // const kyPtNum = outputs[3];
    // const coords = idxs.map(x => divmod(x, mpDim));
    // const featureVec = coords.vstack().T.reshape(2 * kyPtNum, 1);

    return { keypoints, score: 1 };
  }

  // predict(input: tf.Tensor3D): { [key: string]: tf.Tensor3D } {
  predict(input: tf.Tensor3D): { [key: string]: tf.Tensor3D } {
    return tf.tidy(() => {
      const asFloat = toFloatIfInt(input);
      const asBatch = asFloat.expandDims(0);
      const heatmaps4d = this.model.predict(asBatch) as tf.Tensor;

      const heatmapScores = heatmaps4d.squeeze() as tf.Tensor3D;
      // const heatmapScores = heatmaps.sigmoid();
      return {
        heatmapScores
      };
    });
    // return tf.tidy(() => {
    //   const asFloat = toFloatIfInt(input);
    //   const asBatch = asFloat.expandDims(0);
    //   const heatmaps4d = this.model.predict(asBatch) as tf.Tensor;

    //   const heatmaps = heatmaps4d.squeeze() as tf.Tensor3D;
    //   const heatmapScores = heatmaps.sigmoid();
    //   // TODO: I think we could get these direct from the tensor using the GPU
    //   // Get the max x and y values for each keypoint
    //   const keypoints = [];
    //   for (let keypoint = 0; keypoint < 14; keypoint++) {
    //     let max = -1;
    //     let maxX = -1;
    //     let maxY = -1;
    //     for (let x = 0; x < outputW; x++) {
    //       for (const y = 0; y < outputH; y++) {
    //         if (heatmaps[x][y][keypoint] > max) {
    //           max = heatmaps[x][y][keypoint] ;
    //           maxX =x;
    //           maxY =y;
    //       }
    //     }
    //     if (max === -1) {
    //       continue;
    //     }
    //       keypoints += {};
    //   }
    //   const offsets = offsets4d.squeeze() as tf.Tensor3D;
    //   const displacementFwd = displacementFwd4d.squeeze() as tf.Tensor3D;
    //   const displacementBwd = displacementBwd4d.squeeze() as tf.Tensor3D;

    //   return {
    //     heatmapScores,
    //     offsets: offsets as tf.Tensor3D,
    //     displacementFwd: displacementFwd as tf.Tensor3D,
    //     displacementBwd: displacementBwd as tf.Tensor3D
    //   };
    // });
  }

  dispose() {
    this.model.dispose();
  }
}
