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
import { assertValidOutputStride, assertValidResolution } from './mobilenet';
import { decodeSinglePose } from './single_pose/decode_single_pose';
import {
  Pose,
  PosenetInput,
  SinglePersonInterfaceConfig,
  PoseNetInputResolution,
  PoseNetOutputStride
} from './types';
import {
  getInputTensorDimensions,
  padAndResizeTo,
  scaleAndFlipPoses
} from './util';

function toFloatIfInt(input: tf.Tensor3D): tf.Tensor3D {
  return tf.tidy(() => {
    if (input.dtype === 'int32') {
      input = input.toFloat();
    }
    const imageNetMean = tf.tensor([-123.15, -115.9, -103.06]);
    return input.add(imageNetMean);
  });
}

export class ResNet implements BaseModel {
  readonly model: tfconv.GraphModel;
  readonly outputStride: PoseNetOutputStride;

  constructor(model: tfconv.GraphModel, outputStride: PoseNetOutputStride) {
    this.model = model;
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
    this.outputStride = outputStride;
  }

  async estimateSinglePose(
    input: PosenetInput,
    inputResolution: PoseNetInputResolution,
    config: SinglePersonInterfaceConfig
  ): Promise<Pose> {
    const outputStride = this.outputStride;
    assertValidOutputStride(outputStride);
    assertValidResolution(inputResolution, outputStride);

    const [height, width] = getInputTensorDimensions(input);

    const { resized, padding } = padAndResizeTo(input, [
      inputResolution,
      inputResolution
    ]);

    const {
      heatmapScores,
      offsets,
      displacementFwd,
      displacementBwd
    } = this.predict(resized);

    const pose = await decodeSinglePose(heatmapScores, offsets, outputStride);
    const poses = [pose];

    const resultPoses = scaleAndFlipPoses(
      poses,
      [height, width],
      [inputResolution, inputResolution],
      padding,
      config.flipHorizontal
    );

    heatmapScores.dispose();
    offsets.dispose();
    displacementFwd.dispose();
    displacementBwd.dispose();
    resized.dispose();

    return resultPoses[0];
  }

  predict(input: tf.Tensor3D): { [key: string]: tf.Tensor3D } {
    return tf.tidy(() => {
      const asFloat = toFloatIfInt(input);
      const asBatch = asFloat.expandDims(0);
      const [
        displacementFwd4d,
        displacementBwd4d,
        offsets4d,
        heatmaps4d
      ] = this.model.predict(asBatch) as tf.Tensor[];

      const heatmaps = heatmaps4d.squeeze() as tf.Tensor3D;
      const heatmapScores = heatmaps.sigmoid();
      const offsets = offsets4d.squeeze() as tf.Tensor3D;
      const displacementFwd = displacementFwd4d.squeeze() as tf.Tensor3D;
      const displacementBwd = displacementBwd4d.squeeze() as tf.Tensor3D;

      return {
        heatmapScores,
        offsets: offsets as tf.Tensor3D,
        displacementFwd: displacementFwd as tf.Tensor3D,
        displacementBwd: displacementBwd as tf.Tensor3D
      };
    });
  }

  dispose() {
    this.model.dispose();
  }
}
