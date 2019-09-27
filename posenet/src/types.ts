/**
 * @license
 * Copyright 2018 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */

import * as tf from '@tensorflow/tfjs-core';

export declare type Vector2D = {
  y: number;
  x: number;
};

export declare type Part = {
  heatmapX: number;
  heatmapY: number;
  id: number;
};

export declare type PartWithScore = {
  score: number;
  part: Part;
};

export declare type Keypoint = {
  score: number;
  position: Vector2D;
  part: string;
};

export declare type Pose = {
  keypoints: Keypoint[];
  score: number;
};

export type PosenetInput =
  | ImageData
  | HTMLImageElement
  | HTMLCanvasElement
  | HTMLVideoElement
  | tf.Tensor3D;

export type TensorBuffer3D = tf.TensorBuffer<tf.Rank.R3>;

export declare interface Padding {
  top: number;
  bottom: number;
  left: number;
  right: number;
}

/**
 * PoseNet inference is configurable using the following config dictionary.
 *
 * `flipHorizontal`: If the poses should be flipped/mirrored horizontally.
 * This should be set to true for videos where the video is by default flipped
 * horizontally (i.e. a webcam), and you want the poses to be returned in the
 * proper orientation.
 *
 * `inputResolution`:Specifies the size the input image is scaled to before
 * feeding it through the PoseNet model.  The larger the value, more accurate
 * the model at the cost of speed. Set this to a smaller value to increase
 * speed at the cost of accuracy.
 *
 */
export interface InferenceConfig {
  flipHorizontal: boolean;
}

/**
 * Single Person Inference Config
 */
export interface SinglePersonInterfaceConfig extends InferenceConfig {}

/**
 * Multiple Person Inference Config
 *
 * `maxDetections`: Maximum number of returned instance detections per image.
 *
 * `scoreThreshold`: Only return instance detections that have root part
 * score greater or equal to this value. Defaults to 0.5
 *
 * `nmsRadius`: Non-maximum suppression part distance in pixels. It needs
 * to be strictly positive. Two parts suppress each other if they are less
 * than `nmsRadius` pixels away. Defaults to 20.
 **/
export interface MultiPersonInferenceConfig extends InferenceConfig {
  maxDetections?: number;
  scoreThreshold?: number;
  nmsRadius?: number;
}

export type PoseNetInputResolution =
  | 161
  | 193
  | 257
  | 289
  | 321
  | 353
  | 385
  | 417
  | 449
  | 481
  | 513
  | 801
  | 1217;
export type PoseNetOutputStride = 32 | 16 | 8 | 2;
export type PoseNetArchitecture = 'ResNet50' | 'MobileNetV1' | 'CPM';
export type PoseNetDecodingMethod = 'single-person' | 'multi-person';
export type PoseNetQuantBytes = 1 | 2 | 4;
