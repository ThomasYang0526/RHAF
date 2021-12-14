package org.tensorflow.lite.examples.detection;

import androidx.appcompat.app.AppCompatActivity;

import android.content.Intent;
import android.content.res.AssetManager;
import android.graphics.Bitmap;
import android.graphics.BitmapFactory;
import android.graphics.Canvas;
import android.graphics.Color;
import android.graphics.Matrix;
import android.graphics.Paint;
import android.graphics.RectF;
import android.os.Bundle;
import android.os.Environment;
import android.os.Handler;
import android.os.SystemClock;
import android.util.Log;
import android.widget.Button;
import android.widget.ImageView;
import android.widget.Toast;

import org.tensorflow.lite.examples.detection.customview.OverlayView;
import org.tensorflow.lite.examples.detection.env.ImageUtils;
import org.tensorflow.lite.examples.detection.env.Logger;
import org.tensorflow.lite.examples.detection.env.Utils;
import org.tensorflow.lite.examples.detection.tflite.Classifier;
import org.tensorflow.lite.examples.detection.tflite.YoloV4Classifier;
import org.tensorflow.lite.examples.detection.tracking.MultiBoxTracker;

import java.io.BufferedOutputStream;
import java.io.BufferedReader;
import java.io.BufferedWriter;
import java.io.ByteArrayOutputStream;
import java.io.File;
import java.io.IOException;
import java.io.InputStream;
import java.io.InputStreamReader;
import java.io.OutputStream;
import java.io.OutputStreamWriter;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Comparator;
import java.util.HashMap;
import java.util.LinkedList;
import java.util.List;
import java.util.Map;
import java.util.PriorityQueue;
import java.util.Vector;

import com.qualcomm.qti.snpe.FloatTensor;
import com.qualcomm.qti.snpe.NeuralNetwork;
import com.qualcomm.qti.snpe.SNPE;
import com.qualcomm.qti.snpe.Tensor;

import static com.qualcomm.qti.snpe.NeuralNetwork.Runtime.CPU;
import static com.qualcomm.qti.snpe.NeuralNetwork.Runtime.GPU;
import static com.qualcomm.qti.snpe.NeuralNetwork.Runtime.DSP;
import static com.qualcomm.qti.snpe.NeuralNetwork.Runtime.AIP;
import static com.qualcomm.qti.snpe.NeuralNetwork.PerformanceProfile.SUSTAINED_HIGH_PERFORMANCE;
import static org.tensorflow.lite.examples.detection.env.Utils.expit;
import static java.nio.file.StandardOpenOption.CREATE;


public class MainActivity extends AppCompatActivity {

    public static final float MINIMUM_CONFIDENCE_TF_OD_API = 0.5f;
    static NeuralNetwork mNetwork;
    static String mFile_path = Environment.getExternalStorageDirectory().getAbsolutePath();
    private Vector<String> labels = new Vector<String>();

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        // ------- Main -------
        // Setting Model runtime and output layer name.
        try {
            final SNPE.NeuralNetworkBuilder builder = new SNPE.NeuralNetworkBuilder(getApplication())
                    .setRuntimeOrder(DSP)
                    .setCpuFallbackEnabled(true)
                    .setPerformanceProfile(SUSTAINED_HIGH_PERFORMANCE)
                    .setOutputLayers("Reshape_118", "Reshape_129", "Reshape_140", "Reshape_151", "Reshape_162")
                    // Loads a model from DLC file
                    .setModel(new File(mFile_path + "/DCIM/100MEDIA/model_4_quantized.dlc"));
            mNetwork = builder.build();
            System.out.println("HTC_load_model_sucess");
        } catch (final Exception e) {
            System.out.println("HTC_load_model_error " + mFile_path);
        }

        // Input size must be '416x416' and normalize value in '0~1'.
        Map<String, FloatTensor> outputs_seg = new HashMap<>();
        String file_path = Environment.getExternalStorageDirectory().getAbsolutePath();
        float[] rgbBitmapAsFloat_seg;

        Bitmap warped_bitmap = BitmapFactory.decodeFile(mFile_path + "/DCIM/100MEDIA/416x416.jpg");
        FloatTensor tensor_seg = mNetwork.createFloatTensor(mNetwork.getInputTensorsShapes().get("input1"));
        rgbBitmapAsFloat_seg = loadRgbBitmapAsFloat(warped_bitmap);
        tensor_seg.write(rgbBitmapAsFloat_seg, 0, rgbBitmapAsFloat_seg.length);

        final Map<String, FloatTensor> inputs_seg = new HashMap<>();
        inputs_seg.put("input1", tensor_seg);

        long javaExecuteStart;
        long time_check1;
        javaExecuteStart = SystemClock.elapsedRealtime();
        outputs_seg = mNetwork.execute(inputs_seg);
        time_check1 = (SystemClock.elapsedRealtime() - javaExecuteStart);
        Log.i("HTC(timecheck)--exe: ", Double.toString(time_check1));

        // Decoder heatmap and filter object score under score_thres (Post-Process).
        float score_thres = 0.3f;
        float[][] bboxes = Decoder(outputs_seg, score_thres);

        // ------- End -------

        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_main);

        cameraButton = findViewById(R.id.cameraButton);
        detectButton = findViewById(R.id.detectButton);
        imageView = findViewById(R.id.imageView);

        cameraButton1 = findViewById(R.id.button);
        cameraButton2 = findViewById(R.id.button2);
        cameraButton3 = findViewById(R.id.button3);

        cameraButton.setOnClickListener(v -> startActivity(new Intent(MainActivity.this, DetectorActivity.class)));
        cameraButton1.setOnClickListener(v -> startActivity(new Intent(MainActivity.this, DetectorActivityYoloV4Ori.class)));
        cameraButton2.setOnClickListener(v -> startActivity(new Intent(MainActivity.this, DetectorActivityYoloV4Tiny.class)));
        cameraButton3.setOnClickListener(v -> startActivity(new Intent(MainActivity.this, DetectorActivityYoloV4TinyQuantized.class)));

        initBox();
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
        return (original / 255.0f);
//        return (original);
    }

    private float[][] Decoder(Map<String, FloatTensor> outputs_seg, float score_thres){
        // Get output layer array.
        float[] wh = null;
        float[] reg = null;
        float[] raise = null;
        float[] heatmap = null;
        float[] heatmap_maxpool = null;
        for (Map.Entry<String, FloatTensor> output_seg : outputs_seg.entrySet()) {
            if (output_seg.getKey().equals("wh")) {
                FloatTensor outputTensor_seg = output_seg.getValue();
                wh = new float[outputTensor_seg.getSize()];
                outputTensor_seg.read(wh, 0, wh.length);
//                Log.i("HTC(timecheck)--: ", "wh");
            }
            if (output_seg.getKey().equals("reg")) {
                FloatTensor outputTensor_seg = output_seg.getValue();
                reg = new float[outputTensor_seg.getSize()];
                outputTensor_seg.read(reg, 0, reg.length);
//                Log.i("HTC(timecheck)--: ", "reg");
            }
            if (output_seg.getKey().equals("raise")) {
                FloatTensor outputTensor_seg = output_seg.getValue();
                raise = new float[outputTensor_seg.getSize()];
                outputTensor_seg.read(raise, 0, raise.length);
//                Log.i("HTC(timecheck)--: ", "raise");
            }
            if (output_seg.getKey().equals("heatmap")) {
                FloatTensor outputTensor_seg = output_seg.getValue();
                heatmap = new float[outputTensor_seg.getSize()];
                outputTensor_seg.read(heatmap, 0, heatmap.length);
//                Log.i("HTC(timecheck)--: ", "heatmap");
            }
            if (output_seg.getKey().equals("heatmap_maxpool")) {
                FloatTensor outputTensor_seg = output_seg.getValue();
                heatmap_maxpool = new float[outputTensor_seg.getSize()];
                outputTensor_seg.read(heatmap_maxpool, 0, heatmap_maxpool.length);
//                Log.i("HTC(timecheck)--: ", "heatmap_maxpool");
            }
        }
        Log.i("HTC(wh)--: ", Integer.toString(wh.length));
        Log.i("HTC(reg)--: ", Integer.toString(reg.length));
        Log.i("HTC(raise)--: ", Integer.toString(raise.length));
        Log.i("HTC(heatmap)--: ", Integer.toString(heatmap.length));
        Log.i("HTC(heatmap_max)--: ", Integer.toString(heatmap_maxpool.length));

        // Non-maxima location
        for (int idx=0; idx<heatmap_maxpool.length; idx++ ){
            if (heatmap_maxpool[idx] != heatmap[idx]) {
                heatmap_maxpool[idx] = 0.f;
            } else {
                heatmap_maxpool[idx] = (float) (1/(1+Math.exp(-heatmap_maxpool[idx])));
            }
        }

        int person_num = 0;
        int top_k_num = 50;
        int model_inout_size = 416;
        int downsample_ratio = 4;
        int W = model_inout_size/downsample_ratio;
        float[] top_k_score= new float[top_k_num];

        // Doing Top-K
        int[] top_k_indice = getBestKIndices(heatmap_maxpool, top_k_num);

        // Get person object center location and object width/height
        float[] top_k_x= new float[top_k_num];
        float[] top_k_y= new float[top_k_num];
        float[] box_w = new float[top_k_num];
        float[] box_h = new float[top_k_num];
        for (int idx=0; idx < top_k_num; idx++){
            if (heatmap_maxpool[top_k_indice[idx]] > score_thres) {
                top_k_x[idx] = top_k_indice[idx] % W + reg[top_k_indice[idx]*2];
                top_k_y[idx] = top_k_indice[idx] / W + reg[top_k_indice[idx]*2 + 1];
                top_k_score[idx] = heatmap_maxpool[top_k_indice[idx]];
                box_w[idx] = wh[top_k_indice[idx]*2] / 2.0f;
                box_h[idx] = wh[top_k_indice[idx]*2 + 1] / 2.0f;
                person_num += 1;
            }
        }

        /*
        bboxes [idx][xmin, ymin, xmax, ymax, is_raise_hand]
        xmin, ymin, xmax, ymax, ... is ratio value.
        is_raise_hand == 0 means no-raise hand.
        */
        float[][] bboxes = new float[person_num][5];
        if (bboxes.length > 0){
            for (int idx=0; idx<bboxes.length; idx++){
                bboxes[idx][0] = (top_k_x[idx] - box_w[idx])* downsample_ratio;
                bboxes[idx][1] = (top_k_y[idx] - box_h[idx])* downsample_ratio;
                bboxes[idx][2] = (top_k_x[idx] + box_w[idx])* downsample_ratio;
                bboxes[idx][3] = (top_k_y[idx] + box_h[idx])* downsample_ratio;
                if (raise[top_k_indice[idx]*2] > raise[top_k_indice[idx]*2 + 1]){
                    bboxes[idx][4] = 0.0f;
                } else {
                    bboxes[idx][4] = 1.0f;
                }
            }
        }

        return bboxes;
    }

    private int[] getBestKIndices(float[] array, int num) {
        //create sort able array with index and value pair
        IndexValuePair[] pairs = new IndexValuePair[array.length];
        for (int i = 0; i < array.length; i++) {
            pairs[i] = new IndexValuePair(i, array[i]);
        }

        //sort
        Arrays.sort(pairs, new Comparator<IndexValuePair>() {
            public int compare(IndexValuePair o1, IndexValuePair o2) {
                return Float.compare(o2.value, o1.value);
            }
        });

        //extract the indices
        int[] result = new int[num];
        for (int i = 0; i < num; i++) {
            result[i] = pairs[i].index;
        }
        return result;
    }

    private class IndexValuePair {
        private int index;
        private float value;

        public IndexValuePair(int index, float value) {
            this.index = index;
            this.value = value;
        }
    }

    private static final Logger LOGGER = new Logger();

//    public static final int TF_OD_API_INPUT_SIZE = 416;
    public static final int TF_OD_API_INPUT_SIZE = 288;

    private static final boolean TF_OD_API_IS_QUANTIZED = false;

    private static final String TF_OD_API_MODEL_FILE = "yolov4-416-fp32.tflite";
    private static final String TF_OD_API_LABELS_FILE = "file:///android_asset/ocr.txt";

    // Minimum detection confidence to track a detection.
    private static final boolean MAINTAIN_ASPECT = false;
    private Integer sensorOrientation = 90;

    private Classifier detector;

    private Matrix frameToCropTransform;
    private Matrix cropToFrameTransform;
    private MultiBoxTracker tracker;
    private OverlayView trackingOverlay;

    protected int previewWidth = 0;
    protected int previewHeight = 0;

    private Bitmap sourceBitmap;
    private Bitmap cropBitmap;
    private Bitmap warped_bitmap;

    private Button cameraButton, detectButton;
    private Button cameraButton1, cameraButton2, cameraButton3;
    private ImageView imageView;

    private void initBox() {
        previewHeight = TF_OD_API_INPUT_SIZE;
        previewWidth = TF_OD_API_INPUT_SIZE;
        frameToCropTransform = ImageUtils.getTransformationMatrix(previewWidth, previewHeight,TF_OD_API_INPUT_SIZE, TF_OD_API_INPUT_SIZE,
                        sensorOrientation, MAINTAIN_ASPECT);

        cropToFrameTransform = new Matrix();
        frameToCropTransform.invert(cropToFrameTransform);

        tracker = new MultiBoxTracker(this);
        trackingOverlay = findViewById(R.id.tracking_overlay);
        trackingOverlay.addCallback(canvas -> tracker.draw(canvas));

        tracker.setFrameConfiguration(TF_OD_API_INPUT_SIZE, TF_OD_API_INPUT_SIZE, sensorOrientation);

        try {
            detector = YoloV4Classifier.create( getAssets(),
                            TF_OD_API_MODEL_FILE,
                            TF_OD_API_LABELS_FILE,
                            TF_OD_API_IS_QUANTIZED);
        } catch (final IOException e) {
            e.printStackTrace();
            LOGGER.e(e, "Exception initializing classifier!");
            Toast toast = Toast.makeText(getApplicationContext(), "Classifier could not be initialized", Toast.LENGTH_SHORT);
            toast.show();
            finish();
        }
    }

    private void handleResult(Bitmap bitmap, List<Classifier.Recognition> results) {
        final Canvas canvas = new Canvas(bitmap);
        final Paint paint = new Paint();
        paint.setColor(Color.RED);
        paint.setStyle(Paint.Style.STROKE);
        paint.setStrokeWidth(2.0f);

        final List<Classifier.Recognition> mappedRecognitions = new LinkedList<Classifier.Recognition>();

        for (final Classifier.Recognition result : results) {
            final RectF location = result.getLocation();
            if (location != null && result.getConfidence() >= MINIMUM_CONFIDENCE_TF_OD_API) {
                canvas.drawRect(location, paint);
//                cropToFrameTransform.mapRect(location);
//                result.setLocation(location);
//                mappedRecognitions.add(result);
            }
        }
//        tracker.trackResults(mappedRecognitions, new Random().nextInt());
//        trackingOverlay.postInvalidate();
        imageView.setImageBitmap(bitmap);
    }
}
