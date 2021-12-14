package org.tensorflow.lite.examples.detection;

import android.graphics.Bitmap;
import android.graphics.Canvas;
import android.graphics.Matrix;
import android.graphics.RectF;
import android.media.ImageReader;
import android.os.Environment;
import android.os.SystemClock;
import android.util.Log;
import android.util.Size;

import com.qualcomm.qti.snpe.NeuralNetwork;
import com.qualcomm.qti.snpe.SNPE;

import org.tensorflow.lite.examples.detection.customview.OverlayView;
import org.tensorflow.lite.examples.detection.env.ImageUtils;
import org.tensorflow.lite.examples.detection.env.Logger;
import org.tensorflow.lite.examples.detection.tflite.Classifier;
import org.tensorflow.lite.examples.detection.tracking.MultiBoxTracker;

import java.io.BufferedReader;
import java.io.File;
import java.io.IOException;
import java.io.InputStream;
import java.io.InputStreamReader;
import java.util.LinkedList;
import java.util.List;
import java.util.Vector;

import static com.qualcomm.qti.snpe.NeuralNetwork.PerformanceProfile.BURST;
import static com.qualcomm.qti.snpe.NeuralNetwork.PerformanceProfile.SUSTAINED_HIGH_PERFORMANCE;
import static com.qualcomm.qti.snpe.NeuralNetwork.Runtime.GPU_FLOAT16;
import static com.qualcomm.qti.snpe.NeuralNetwork.Runtime.GPU;

public class DetectorActivityYoloV4Tiny extends CameraActivity implements ImageReader.OnImageAvailableListener {

    private static final Logger LOGGER = new Logger();
    private static final int TF_OD_API_INPUT_SIZE = 288;
    private static final String TF_OD_API_LABELS_FILE = "file:///android_asset/coco_thomas.txt";

    private static final float MINIMUM_CONFIDENCE_TF_OD_API = 0.5f;
    private static final boolean MAINTAIN_ASPECT = false;
    private static final Size DESIRED_PREVIEW_SIZE = new Size(640, 480);
    private static final boolean SAVE_PREVIEW_BITMAP = false;
    OverlayView trackingOverlay;
    private Integer sensorOrientation;

    private long lastProcessingTimeMs;
    private Bitmap rgbFrameBitmap = null;
    private Bitmap croppedBitmap = null;
    private Bitmap cropCopyBitmap = null;

    private boolean computingDetection = false;

    private long timestamp = 0;

    private Matrix frameToCropTransform;
    private Matrix cropToFrameTransform;

    private MultiBoxTracker tracker;

    private NeuralNetwork mNetwork;
    private String mFile_path = Environment.getExternalStorageDirectory().getAbsolutePath();
    private Vector<String> mlabels = new Vector<String>();
    List<Classifier.Recognition> mresults = new LinkedList<Classifier.Recognition>();

    @Override
    public void onPreviewSizeChosen(final Size size, final int rotation) {


        tracker = new MultiBoxTracker(this);

        int cropSize = TF_OD_API_INPUT_SIZE;

        // YoloV4 Tiny Onnx Size Model.
        try {
            final SNPE.NeuralNetworkBuilder builder = new SNPE.NeuralNetworkBuilder(getApplication())
                    .setRuntimeOrder(GPU)
                    .setPerformanceProfile(SUSTAINED_HIGH_PERFORMANCE)
                    .setOutputLayers("conv2d_17/Conv2D", "conv2d_20/Conv2D")
//                    .setOutputLayers("030_convolutional", "037_convolutional")
                    // Loads a model from DLC file
                    .setModel(new File(mFile_path + "/DCIM/100MEDIA/test3.dlc"));
//                    .setModel(new File(mFile_path + "/DCIM/100MEDIA/yolov4_tiny_onnx_288.dlc"));
            mNetwork = builder.build();
            System.out.println("Thomas_load_model_sucess");
        } catch (final Exception e) {
            System.out.println("Thomas_load_model_error");
        }

//        try {
//            final SNPE.NeuralNetworkBuilder builder = new SNPE.NeuralNetworkBuilder(getApplication())
//                    .setRuntimeOrder(GPU)
//                    .setPerformanceProfile(BURST)
//                    .setCpuFallbackEnabled(true)
//                    .setOutputLayers("detector/yolo-v4-tiny/Conv_17/Conv2D",
//                                     "detector/yolo-v4-tiny/Conv_20/Conv2D")
//                    // Loads a model from DLC file
//                    .setModel(new File(mFile_path + "/DCIM/100MEDIA/yolov4-tiny.dlc"));
//            mNetwork = builder.build();
//            System.out.println("Thomas_load_model_sucess");
//        } catch (final Exception e) {
//            System.out.println("Thomas_load_model_error");
//        }

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

        previewWidth = size.getWidth();
        previewHeight = size.getHeight();

        sensorOrientation = rotation - getScreenOrientation();
        LOGGER.i("Camera orientation relative to screen canvas: %d", sensorOrientation);

        LOGGER.i("Initializing at size %dx%d", previewWidth, previewHeight);
        rgbFrameBitmap = Bitmap.createBitmap(previewWidth, previewHeight, Bitmap.Config.ARGB_8888);
        croppedBitmap = Bitmap.createBitmap(cropSize, cropSize, Bitmap.Config.ARGB_8888);

        frameToCropTransform = ImageUtils.getTransformationMatrix(previewWidth, previewHeight, cropSize, cropSize, sensorOrientation, MAINTAIN_ASPECT);

        cropToFrameTransform = new Matrix();
        frameToCropTransform.invert(cropToFrameTransform);

        trackingOverlay = (OverlayView) findViewById(R.id.tracking_overlay);
        trackingOverlay.addCallback(
                new OverlayView.DrawCallback() {
                    @Override
                    public void drawCallback(final Canvas canvas) {
                        tracker.draw(canvas);
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
                        RunSnpeModel yoloV4 = new RunSnpeModel();
                        final List<Classifier.Recognition> results = yoloV4.run(mNetwork, croppedBitmap, mlabels, true);
                        lastProcessingTimeMs = SystemClock.uptimeMillis() - startTime;

                        Log.e("CHECK", "run: " + results.size());

                        final List<Classifier.Recognition> mappedRecognitions = new LinkedList<Classifier.Recognition>();

                        for (final Classifier.Recognition result : results) {
                            final RectF location = result.getLocation();
                            if (location != null && result.getConfidence() >= MINIMUM_CONFIDENCE_TF_OD_API && result.getTitle().equals("person")) {
//                                canvas.drawRect(location, paint);
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
                                        showCropInfo(TF_OD_API_INPUT_SIZE + "x" + TF_OD_API_INPUT_SIZE);
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

    @Override
    protected void setUseNNAPI(final boolean isChecked) {
    }

    @Override
    protected void setNumThreads(final int numThreads) {
    }
}
