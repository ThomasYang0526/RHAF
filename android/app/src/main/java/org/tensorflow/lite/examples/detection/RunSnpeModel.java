package org.tensorflow.lite.examples.detection;

import android.content.res.AssetManager;
import android.graphics.Bitmap;
import android.graphics.Canvas;
import android.graphics.Color;
import android.graphics.Paint;
import android.graphics.RectF;
import android.os.Bundle;
import android.os.Environment;
import android.os.SystemClock;
import android.util.Log;

import com.qualcomm.qti.snpe.FloatTensor;
import com.qualcomm.qti.snpe.NeuralNetwork;
import com.qualcomm.qti.snpe.SNPE;
import com.qualcomm.qti.snpe.Tensor;

import org.tensorflow.lite.examples.detection.tflite.Classifier;
import org.tensorflow.lite.examples.detection.env.Logger;

import java.io.BufferedReader;
import java.io.File;
import java.io.IOException;
import java.io.InputStream;
import java.io.InputStreamReader;
import java.util.ArrayList;
import java.util.Comparator;
import java.util.HashMap;
import java.util.LinkedList;
import java.util.List;
import java.util.Map;
import java.util.PriorityQueue;
import java.util.Vector;
import android.content.Context;
import android.util.Size;


import static com.qualcomm.qti.snpe.NeuralNetwork.PerformanceProfile.BURST;
import static com.qualcomm.qti.snpe.NeuralNetwork.Runtime.GPU;
import static org.tensorflow.lite.examples.detection.env.Utils.expit;

public class RunSnpeModel {

    private static final Logger LOGGER = new Logger();
    private static final String TF_OD_API_LABELS_FILE = "file:///android_asset/coco.txt";
    private boolean mIstiny;

//    public ArrayList<Classifier.Recognition> run(NeuralNetwork mNetwork,
//                                                 Bitmap croppedBitmap,
//                                                 Vector<String> labels,
//                                                 String InputTensorName,
//                                                 String OutputTensorName0,
//                                                 String OutputTensorName1,
//                                                 String OutputTensorName2,
//                                                 boolean isTiny) {

    public ArrayList<Classifier.Recognition> run(NeuralNetwork mNetwork,
                                                    Bitmap croppedBitmap,
                                                    Vector<String> labels,
                                                    boolean isTiny) {
        mIstiny = isTiny;
        long ModelInferenceTimeMs, PostProTimeMs;

        FloatTensor tensor_seg;
        if(isTiny) {
            tensor_seg = mNetwork.createFloatTensor(mNetwork.getInputTensorsShapes().get("input_1:0"));
        } else {
            tensor_seg = mNetwork.createFloatTensor(mNetwork.getInputTensorsShapes().get("inputs:0"));
        }
        float[] rgbBitmapAsFloat_seg;
        rgbBitmapAsFloat_seg = loadRgbBitmapAsFloat(croppedBitmap);
        tensor_seg.write(rgbBitmapAsFloat_seg, 0, rgbBitmapAsFloat_seg.length);

        final Map<String, FloatTensor> inputs_seg = new HashMap<>();
        Map<String, FloatTensor> outputs_seg;
        long javaExecuteStart;
        long time_check;
        if(isTiny) {
            inputs_seg.put("input_1:0", tensor_seg);
        } else {
            inputs_seg.put("inputs:0", tensor_seg);
        }
        long startTime = SystemClock.elapsedRealtime();
        outputs_seg = mNetwork.execute(inputs_seg);
        long lastProcessingTimeMs = SystemClock.elapsedRealtime() - startTime;
        Log.i("Thomas(01)--model", Double.toString(lastProcessingTimeMs));
        releaseTensors(inputs_seg);
        ModelInferenceTimeMs = lastProcessingTimeMs;

        float[] detection_1 = null;
        float[] detection_2 = null;
        float[] detection_3 = null;

//        for (Map.Entry<String, FloatTensor> output_seg : outputs_seg.entrySet()) {
//            if (output_seg.getKey().equals("139_convolutional")) {
//                FloatTensor outputTensor_seg = output_seg.getValue();
//                detection_1 = new float[outputTensor_seg.getSize()];
//                outputTensor_seg.read(detection_1, 0, detection_1.length);
//                System.out.println("Thomas_037_convolutional");
//            }
//            if (output_seg.getKey().equals("150_convolutional")) {
//                FloatTensor outputTensor_seg = output_seg.getValue();
//                detection_2 = new float[outputTensor_seg.getSize()];
//                outputTensor_seg.read(detection_2, 0, detection_2.length);
//                System.out.println("Thomas_030_convolutional");
//            }
//            if (output_seg.getKey().equals("161_convolutional")) {
//                FloatTensor outputTensor_seg = output_seg.getValue();
//                detection_2 = new float[outputTensor_seg.getSize()];
//                outputTensor_seg.read(detection_2, 0, detection_2.length);
//                System.out.println("Thomas_027_convolutional");
//            }
//        }

        if (isTiny) {

            for (Map.Entry<String, FloatTensor> output_seg : outputs_seg.entrySet()) {
                if (output_seg.getKey().equals("conv2d_20/BiasAdd:0")) {
                    FloatTensor outputTensor_seg = output_seg.getValue();
                    detection_1 = new float[outputTensor_seg.getSize()];
                    outputTensor_seg.read(detection_1, 0, detection_1.length);
//                    System.out.println("Thomas_037_convolutional");
                }
                if (output_seg.getKey().equals("conv2d_17/BiasAdd:0")) {
                    FloatTensor outputTensor_seg = output_seg.getValue();
                    detection_2 = new float[outputTensor_seg.getSize()];
                    outputTensor_seg.read(detection_2, 0, detection_2.length);
//                    System.out.println("Thomas_030_convolutional");
                }
            }

//            for (Map.Entry<String, FloatTensor> output_seg : outputs_seg.entrySet()) {
//                if (output_seg.getKey().equals("detector/yolo-v4-tiny/detect_2:0")) {
//                    FloatTensor outputTensor_seg = output_seg.getValue();
//                    detection_1 = new float[outputTensor_seg.getSize()];
//                    outputTensor_seg.read(detection_1, 0, detection_1.length);
//                    System.out.println("Thomas_037_convolutional");
//                }
//                if (output_seg.getKey().equals("detector/yolo-v4-tiny/detect_1:0")) {
//                    FloatTensor outputTensor_seg = output_seg.getValue();
//                    detection_2 = new float[outputTensor_seg.getSize()];
//                    outputTensor_seg.read(detection_2, 0, detection_2.length);
//                    System.out.println("Thomas_030_convolutional");
//                }
//            }
        } else {
            for (Map.Entry<String, FloatTensor> output_seg : outputs_seg.entrySet()) {
                if (output_seg.getKey().equals("detector/yolo-v4/detect_1:0")) {
                    FloatTensor outputTensor_seg = output_seg.getValue();
                    detection_1 = new float[outputTensor_seg.getSize()];
                    outputTensor_seg.read(detection_1, 0, detection_1.length);
//                    System.out.println("Thomas_037_convolutional");
                }
                if (output_seg.getKey().equals("detector/yolo-v4/detect_2:0")) {
                    FloatTensor outputTensor_seg = output_seg.getValue();
                    detection_2 = new float[outputTensor_seg.getSize()];
                    outputTensor_seg.read(detection_2, 0, detection_2.length);
//                    System.out.println("Thomas_030_convolutional");
                }
                if (output_seg.getKey().equals("detector/yolo-v4/detect_3:0")) {
                    FloatTensor outputTensor_seg = output_seg.getValue();
                    detection_3 = new float[outputTensor_seg.getSize()];
                    outputTensor_seg.read(detection_3, 0, detection_3.length);
//                    System.out.println("Thomas_027_convolutional");
                }
            }
        }

        javaExecuteStart = SystemClock.elapsedRealtime();
        int NUM_BOXES_PER_BLOCK = 3;
        float getObjThresh = 0.5f;
        int INPUT_SIZE = 288;

        int[] gridWidth;
        final int[][] MASKS;
        final int[] ANCHORS;
        if (isTiny) {
            gridWidth = new int[]{18, 9};
//            MASKS = new int[][]{{1, 2, 3}, {3, 4, 5}};
//            ANCHORS = new int[]{10, 14, 23, 27, 37, 58, 81, 82, 135, 169, 344, 319};
            MASKS = new int[][]{{0, 1, 2}, {3, 4, 5}};
            ANCHORS = new int[]{18,74, 54,105, 63,46, 82,192, 136,86, 174,187};
        } else {
            gridWidth = new int[]{36, 18, 9};
            MASKS = new int[][]{{0, 1, 2}, {3, 4, 5}, {6, 7, 8}};
            ANCHORS = new int[]{12, 16, 19, 36, 40, 28, 36, 75, 76, 55, 72, 146, 142, 110, 192, 243, 459, 401};
        }

        ArrayList<Classifier.Recognition> detections = new ArrayList<Classifier.Recognition>();


        for (int bboxes = 0; bboxes<gridWidth.length; bboxes++){
            startTime = SystemClock.elapsedRealtime();
            float[] out = new float[]{0};
            if(bboxes == 0){
                out = detection_1;
            } else if(bboxes==1){
                out = detection_2;
            } else if(bboxes==2){
                out = detection_3;
            }

            lastProcessingTimeMs = SystemClock.elapsedRealtime() - startTime;
            Log.i("Thomas(02)------post__" + bboxes, Double.toString(lastProcessingTimeMs));

            startTime = SystemClock.elapsedRealtime();
            int offsetY, offsetX, offsetB, offset;
            for (int y = 0; y < gridWidth[bboxes]; ++y) {
                offsetY = (gridWidth[bboxes] * (NUM_BOXES_PER_BLOCK * (labels.size() + 5)))*y;
                for (int x = 0; x < gridWidth[bboxes]; ++x) {
                    offsetX = (NUM_BOXES_PER_BLOCK * (labels.size() + 5)) * x;
                    for (int b = 0; b < NUM_BOXES_PER_BLOCK; ++b) {
                        offsetB = (labels.size() + 5) * b;
                        offset = offsetY + offsetX + offsetB;

                        final float confidence = expit(out[offset + 4]);
                        int detectedClass = -1;
                        float maxClass = 0;
                        float sum_exp = 0;
                        float exp_con = 0;
                        final float[] classes = new float[labels.size()];

//                        for (int c = 0; c < labels.size(); ++c) {
//                            exp_con = (float)Math.exp(out[offset + 5 + c]);
//                            classes[c] = exp_con;
//                            sum_exp += exp_con;
//                            if (exp_con > maxClass) {
//                                detectedClass = c;
//                                maxClass = exp_con;
//                            }
//                        }

                        final float confidenceInClass = maxClass * confidence;
//                        if (confidenceInClass > getObjThresh) {
                        if (confidence > getObjThresh) {

                            detectedClass = 0;
                            final float xPos = (x + expit(out[offset + 0])) * (1.0f * INPUT_SIZE / gridWidth[bboxes]);
                            final float yPos = (y + expit(out[offset + 1])) * (1.0f * INPUT_SIZE / gridWidth[bboxes]);

                            final float w = (float) (Math.exp(out[offset + 2]) * ANCHORS[2 * MASKS[bboxes][b]]);
                            final float h = (float) (Math.exp(out[offset + 3]) * ANCHORS[2 * MASKS[bboxes][b] + 1]);

                            float scale = 2.0f;
                            final RectF rect =
                                    new RectF(Math.max(0, xPos - w / scale),
                                              Math.max(0, yPos - h / scale),
                                              Math.min(croppedBitmap.getWidth() - 1, xPos + w / scale),
                                              Math.min(croppedBitmap.getHeight() - 1, yPos + h / scale));
//                            detections.add(new Classifier.Recognition("" + offset, labels.get(detectedClass), confidenceInClass, rect, detectedClass));
                            detections.add(new Classifier.Recognition("" + offset, labels.get(detectedClass), confidence, rect, detectedClass));
                        }
                    }
                }
            }
            lastProcessingTimeMs = SystemClock.elapsedRealtime() - startTime;
            Log.i("Thomas(03)------post__" + bboxes, Double.toString(lastProcessingTimeMs));
        }

        final ArrayList<Classifier.Recognition> recognitions = nms(detections, labels);
        PostProTimeMs = lastProcessingTimeMs;

        return recognitions;
    }

    float[] loadRgbBitmapAsFloat(Bitmap image) {
        final int[] pixels = new int[image.getWidth() * image.getHeight()];
        image.getPixels(pixels, 0, image.getWidth(), 0, 0,
                image.getWidth(), image.getHeight());

        final float[] pixelsBatched = new float[pixels.length * 3];
        for (int y = 0; y < image.getHeight(); y++) {
            for (int x = 0; x < image.getWidth(); x++) {
                final int idx = y * image.getWidth() + x;
                final int batchIdx = idx * 3;

                final float[] rgb = extractColorChannels(pixels[idx]);
                pixelsBatched[batchIdx]     = rgb[0];
                pixelsBatched[batchIdx + 1] = rgb[1];
                pixelsBatched[batchIdx + 2] = rgb[2];
            }
        }
        return pixelsBatched;
    }

    private float[] extractColorChannels(int pixel) {
//        String modelName = mModel.name;
        float b = ((pixel)       & 0xFF);
        float g = ((pixel >>  8) & 0xFF);
        float r = ((pixel >> 16) & 0xFF);
        return new float[] {preProcess(r), preProcess(g), preProcess(b)};
    }

    private float preProcess(float original) {
//        String modelName = mModel.name;
        if(mIstiny) {
            return (original / 255.0f);
        } else {
            return (original);
        }
    }

    private final void releaseTensors(Map<String, ? extends Tensor>... tensorMaps) {
        for (Map<String, ? extends Tensor> tensorMap: tensorMaps) {
            for (Tensor tensor: tensorMap.values()) {
                tensor.release();
            }
        }
    }

    protected float box_iou(RectF a, RectF b) {
        return box_intersection(a, b) / box_union(a, b);
    }

    protected float box_intersection(RectF a, RectF b) {
        float w = overlap((a.left + a.right) / 2, a.right - a.left,
                (b.left + b.right) / 2, b.right - b.left);
        float h = overlap((a.top + a.bottom) / 2, a.bottom - a.top,
                (b.top + b.bottom) / 2, b.bottom - b.top);
        if (w < 0 || h < 0) return 0;
        float area = w * h;
        return area;
    }

    protected float box_union(RectF a, RectF b) {
        float i = box_intersection(a, b);
        float u = (a.right - a.left) * (a.bottom - a.top) + (b.right - b.left) * (b.bottom - b.top) - i;
        return u;
    }

    protected float overlap(float x1, float w1, float x2, float w2) {
        float l1 = x1 - w1 / 2;
        float l2 = x2 - w2 / 2;
        float left = l1 > l2 ? l1 : l2;
        float r1 = x1 + w1 / 2;
        float r2 = x2 + w2 / 2;
        float right = r1 < r2 ? r1 : r2;
        return right - left;
    }

    protected ArrayList<Classifier.Recognition> nms(ArrayList<Classifier.Recognition> list, Vector<String> labels) {
        ArrayList<Classifier.Recognition> nmsList = new ArrayList<Classifier.Recognition>();

        float mNmsThresh = 0.5f;
        for (int k = 0; k < labels.size(); k++) {
            //1.find max confidence per class
            PriorityQueue<Classifier.Recognition> pq =
                    new PriorityQueue<Classifier.Recognition>(
                            50,
                            new Comparator<Classifier.Recognition>() {
                                @Override
                                public int compare(final Classifier.Recognition lhs, final Classifier.Recognition rhs) {
                                    // Intentionally reversed to put high confidence at the head of the queue.
                                    return Float.compare(rhs.getConfidence(), lhs.getConfidence());
                                }
                            });

            for (int i = 0; i < list.size(); ++i) {
                if (list.get(i).getDetectedClass() == k) {
                    pq.add(list.get(i));
                }
            }

            //2.do non maximum suppression
            while (pq.size() > 0) {
                //insert detection with max confidence
                Classifier.Recognition[] a = new Classifier.Recognition[pq.size()];
                Classifier.Recognition[] detections = pq.toArray(a);
                Classifier.Recognition max = detections[0];
                nmsList.add(max);

                pq.clear();

                for (int j = 1; j < detections.length; j++) {
                    Classifier.Recognition detection = detections[j];
                    RectF b = detection.getLocation();
                    if (box_iou(max.getLocation(), b) < mNmsThresh) {
                        pq.add(detection);
                    }
                }
            }
        }
        return nmsList;
    }

}
