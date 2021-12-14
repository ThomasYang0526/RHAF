/*
 * Copyright 2019 The TensorFlow Authors. All Rights Reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *       http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

package org.tensorflow.lite.examples.detection;

import android.graphics.Bitmap;
import android.graphics.Bitmap.Config;
import android.graphics.Canvas;
import android.graphics.Color;
import android.graphics.Matrix;
import android.graphics.Paint;
import android.graphics.Paint.Style;
import android.graphics.RectF;
import android.graphics.Typeface;
import android.media.ImageReader.OnImageAvailableListener;
import android.os.Environment;
import android.os.SystemClock;
import android.util.Log;
import android.util.Size;
import android.util.TypedValue;
import android.widget.Toast;

import com.qualcomm.qti.snpe.NeuralNetwork;
import com.qualcomm.qti.snpe.SNPE;

import java.io.BufferedReader;
import java.io.File;
import java.io.IOException;
import java.io.InputStream;
import java.io.InputStreamReader;
import java.util.LinkedList;
import java.util.List;
import java.util.Vector;

import org.tensorflow.lite.examples.detection.customview.OverlayView;
import org.tensorflow.lite.examples.detection.customview.OverlayView.DrawCallback;
import org.tensorflow.lite.examples.detection.env.BorderedText;
import org.tensorflow.lite.examples.detection.env.ImageUtils;
import org.tensorflow.lite.examples.detection.env.Logger;
import org.tensorflow.lite.examples.detection.tflite.Classifier;
import org.tensorflow.lite.examples.detection.tflite.YoloV4Classifier;
import org.tensorflow.lite.examples.detection.tracking.MultiBoxTracker;

import static com.qualcomm.qti.snpe.NeuralNetwork.PerformanceProfile.BURST;
import static com.qualcomm.qti.snpe.NeuralNetwork.Runtime.GPU_FLOAT16;

/**
 * An activity that uses a TensorFlowMultiBoxDetector and ObjectTracker to detect and then track
 * objects.
 */
public class DetectorActivity extends CameraActivity implements OnImageAvailableListener {

    private static final Logger LOGGER = new Logger();

//    private static final int TF_OD_API_INPUT_SIZE = 416;
    private static final int TF_OD_API_INPUT_SIZE = 288;
    private static final boolean TF_OD_API_IS_QUANTIZED = false;
    private static final String TF_OD_API_MODEL_FILE = "yolov4-416-fp32.tflite";

    private static final String TF_OD_API_LABELS_FILE = "file:///android_asset/coco.txt";

    private static final DetectorMode MODE = DetectorMode.TF_OD_API;
    private static final float MINIMUM_CONFIDENCE_TF_OD_API = 0.5f;
    private static final boolean MAINTAIN_ASPECT = false;
    private static final Size DESIRED_PREVIEW_SIZE = new Size(640, 480);
    private static final boolean SAVE_PREVIEW_BITMAP = false;
    private static final float TEXT_SIZE_DIP = 10;
    OverlayView trackingOverlay;
    private Integer sensorOrientation;

    private Classifier detector;

    private long lastProcessingTimeMs;
    private Bitmap rgbFrameBitmap = null;
    private Bitmap croppedBitmap = null;
    private Bitmap cropCopyBitmap = null;

    private boolean computingDetection = false;

    private long timestamp = 0;

    private Matrix frameToCropTransform;
    private Matrix cropToFrameTransform;

    private MultiBoxTracker tracker;

    private BorderedText borderedText;

    private NeuralNetwork mNetwork;
    private String mFile_path = Environment.getExternalStorageDirectory().getAbsolutePath();
    private Vector<String> mlabels = new Vector<String>();
    List<Classifier.Recognition> mresults = new LinkedList<Classifier.Recognition>();

    @Override
    public void onPreviewSizeChosen(final Size size, final int rotation) {
        final float textSizePx = TypedValue.applyDimension(TypedValue.COMPLEX_UNIT_DIP, TEXT_SIZE_DIP, getResources().getDisplayMetrics());
        borderedText = new BorderedText(textSizePx);
        borderedText.setTypeface(Typeface.MONOSPACE);

        tracker = new MultiBoxTracker(this);

        int cropSize = TF_OD_API_INPUT_SIZE;

        try {
            detector =
                    YoloV4Classifier.create(
                            getAssets(),
                            TF_OD_API_MODEL_FILE,
                            TF_OD_API_LABELS_FILE,
                            TF_OD_API_IS_QUANTIZED);
//            detector = TFLiteObjectDetectionAPIModel.create(
//                    getAssets(),
//                    TF_OD_API_MODEL_FILE,
//                    TF_OD_API_LABELS_FILE,
//                    TF_OD_API_INPUT_SIZE,
//                    TF_OD_API_IS_QUANTIZED);
            cropSize = TF_OD_API_INPUT_SIZE;
        } catch (final IOException e) {
            e.printStackTrace();
            LOGGER.e(e, "Exception initializing classifier!");
            Toast toast =
                    Toast.makeText(
                            getApplicationContext(), "Classifier could not be initialized", Toast.LENGTH_SHORT);
            toast.show();
            finish();
        }

//        try {
//            final SNPE.NeuralNetworkBuilder builder = new SNPE.NeuralNetworkBuilder(getApplication())
//                    .setRuntimeOrder(GPU)
//                    .setPerformanceProfile(BURST)
//                    .setOutputLayers("030_convolutional", "037_convolutional")
//                    // Loads a model from DLC file
//                    .setModel(new File(mFile_path + "/DCIM/100MEDIA/yolov4_tiny_onnx_288.dlc"));
////                    .setModel(new File(mFile_path + "/DCIM/100MEDIA/yolov4_tiny_onnx_288_quantized.dlc"));
//            mNetwork = builder.build();
//            System.out.println("Thomas_load_model_sucess");
//        } catch (final Exception e) {
//            System.out.println("Thomas_load_model_error");
//        }

//        try {
//            final SNPE.NeuralNetworkBuilder builder = new SNPE.NeuralNetworkBuilder(getApplication())
//                    .setCpuFallbackEnabled(true)
//                    .setRuntimeOrder(DSP)
//                    .setPerformanceProfile(BURST)
//                    .setOutputLayers("139_convolutional", "150_convolutional", "161_convolutional")
//                    // Loads a model from DLC file
//                    .setModel(new File(mFile_path + "/DCIM/100MEDIA/yolov4_onnx_quantized.dlc"));
//            mNetwork = builder.build();
//            System.out.println("Thomas_load_model_sucess");
//        } catch (final Exception e) {
//            System.out.println("Thomas_load_model_error");
//        }

//        try {
//            final SNPE.NeuralNetworkBuilder builder = new SNPE.NeuralNetworkBuilder(getApplication())
//                    .setRuntimeOrder(GPU)
//                    .setPerformanceProfile(BURST)
//                    .setCpuFallbackEnabled(true)
//                    .setOutputLayers("detector/yolo-v4-tiny/Conv_20/Conv2D","detector/yolo-v4-tiny/Conv_17/Conv2D")
//                    // Loads a model from DLC file
//                    .setModel(new File(mFile_path + "/DCIM/100MEDIA/yolov4-tiny.dlc"));
//            mNetwork = builder.build();
//            System.out.println("Thomas_load_model_sucess");
//        } catch (final Exception e) {
//            System.out.println("Thomas_load_model_error");
//        }

        try {
            final SNPE.NeuralNetworkBuilder builder = new SNPE.NeuralNetworkBuilder(getApplication())
                    .setRuntimeOrder(GPU_FLOAT16)
                    .setPerformanceProfile(BURST)
                    .setCpuFallbackEnabled(true)
                    .setOutputLayers("detector/yolo-v4/Conv_1/Conv2D",
                            "detector/yolo-v4/Conv_9/Conv2D",
                            "detector/yolo-v4/Conv_17/Conv2D")
                    // Loads a model from DLC file
                    .setModel(new File(mFile_path + "/DCIM/100MEDIA/yolov4.dlc"));
            mNetwork = builder.build();
            System.out.println("Thomas_load_model_sucess");
        } catch (final Exception e) {
            System.out.println("Thomas_load_model_error");
        }


        if(mlabels.size() != 80) {
            try {
                String actualFilename = TF_OD_API_LABELS_FILE.split("file:///android_asset/")[1];
                InputStream labelsInput = getAssets().open(actualFilename);
                BufferedReader br = new BufferedReader(new InputStreamReader(labelsInput));
                String line;
                while ((line = br.readLine()) != null) {
                    LOGGER.w(line);
                    mlabels.add(line);
                }
                br.close();
            } catch (final IOException e) {
                System.out.println("Thomas_load_model_error");
            }
        }

        previewWidth = size.getWidth();
        previewHeight = size.getHeight();

        sensorOrientation = rotation - getScreenOrientation();
        LOGGER.i("Camera orientation relative to screen canvas: %d", sensorOrientation);

        LOGGER.i("Initializing at size %dx%d", previewWidth, previewHeight);
        rgbFrameBitmap = Bitmap.createBitmap(previewWidth, previewHeight, Config.ARGB_8888);
        croppedBitmap = Bitmap.createBitmap(cropSize, cropSize, Config.ARGB_8888);

        frameToCropTransform =
                ImageUtils.getTransformationMatrix(
                        previewWidth, previewHeight,
                        cropSize, cropSize,
                        sensorOrientation, MAINTAIN_ASPECT);

        cropToFrameTransform = new Matrix();
        frameToCropTransform.invert(cropToFrameTransform);

        trackingOverlay = (OverlayView) findViewById(R.id.tracking_overlay);
        trackingOverlay.addCallback(
                new DrawCallback() {
                    @Override
                    public void drawCallback(final Canvas canvas) {
                        tracker.draw(canvas);
//                        tracker.drawT(canvas, mresults);
                        if (isDebug()) {
                            tracker.drawDebug(canvas);
                        }
                    }
                });

        tracker.setFrameConfiguration(previewWidth, previewHeight, sensorOrientation);
    }

    @Override
    protected void processImage() {
        ++timestamp;
        final long currTimestamp = timestamp;
        trackingOverlay.postInvalidate();

        // No mutex needed as this method is not reentrant.
        if (computingDetection) {
            readyForNextImage();
            return;
        }
        computingDetection = true;
        LOGGER.i("Preparing image " + currTimestamp + " for detection in bg thread.");

        rgbFrameBitmap.setPixels(getRgbBytes(), 0, previewWidth, 0, 0, previewWidth, previewHeight);

        readyForNextImage();

        final Canvas canvas = new Canvas(croppedBitmap);
        canvas.drawBitmap(rgbFrameBitmap, frameToCropTransform, null);
        // For examining the actual TF input.
        if (SAVE_PREVIEW_BITMAP) {
            ImageUtils.saveBitmap(croppedBitmap);
        }

        runInBackground(
                new Runnable() {
                    @Override
                    public void run() {

                        LOGGER.i("Running detection on image " + currTimestamp);
                        final long startTime = SystemClock.uptimeMillis();
//                        final List<Classifier.Recognition> results = detector.recognizeImage(croppedBitmap);
                        RunSnpeModel yoloV4 = new RunSnpeModel();
                        final List<Classifier.Recognition> results = yoloV4.run(mNetwork, croppedBitmap, mlabels, false);
                        lastProcessingTimeMs = SystemClock.uptimeMillis() - startTime;

                        Log.e("CHECK", "run: " + results.size());

                        cropCopyBitmap = Bitmap.createBitmap(croppedBitmap);
                        final Canvas canvas = new Canvas(cropCopyBitmap);
                        final Paint paint = new Paint();
                        paint.setColor(Color.RED);
                        paint.setStyle(Style.STROKE);
                        paint.setStrokeWidth(2.0f);

                        float minimumConfidence = MINIMUM_CONFIDENCE_TF_OD_API;
                        switch (MODE) {
                            case TF_OD_API:
                                minimumConfidence = MINIMUM_CONFIDENCE_TF_OD_API;
                                break;
                        }


                        final List<Classifier.Recognition> mappedRecognitions = new LinkedList<Classifier.Recognition>();

                        for (final Classifier.Recognition result : results) {
                            final RectF location = result.getLocation();
                            if (location != null && result.getConfidence() >= minimumConfidence && result.getTitle().equals("person")) {
                                canvas.drawRect(location, paint);
                                cropToFrameTransform.mapRect(location);
                                result.setLocation(location);
                                mappedRecognitions.add(result);
                            }
                        }

                        mresults = results;
                        tracker.trackResults(mappedRecognitions, currTimestamp);
                        trackingOverlay.postInvalidate();

                        computingDetection = false;

                        runOnUiThread(
                                new Runnable() {
                                    @Override
                                    public void run() {
                                        showFrameInfo(previewWidth + "x" + previewHeight);
                                        showCropInfo(cropCopyBitmap.getWidth() + "x" + cropCopyBitmap.getHeight());
                                        showInference(lastProcessingTimeMs + "ms");
                                    }
                                });
                    }
                });
    }

    @Override
    protected int getLayoutId() {
        return R.layout.tfe_od_camera_connection_fragment_tracking;
    }

    @Override
    protected Size getDesiredPreviewFrameSize() {
        return DESIRED_PREVIEW_SIZE;
    }

    // Which detection model to use: by default uses Tensorflow Object Detection API frozen
    // checkpoints.
    private enum DetectorMode {
        TF_OD_API;
    }

    @Override
    protected void setUseNNAPI(final boolean isChecked) {
        runInBackground(() -> detector.setUseNNAPI(isChecked));
    }

    @Override
    protected void setNumThreads(final int numThreads) {
        runInBackground(() -> detector.setNumThreads(numThreads));
    }
}
