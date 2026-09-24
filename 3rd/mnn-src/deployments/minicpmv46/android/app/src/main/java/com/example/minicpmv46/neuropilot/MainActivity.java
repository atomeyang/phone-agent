package com.example.minicpmv46.neuropilot;

import android.app.Activity;
import android.content.Intent;
import android.content.res.AssetFileDescriptor;
import android.graphics.Bitmap;
import android.graphics.Canvas;
import android.graphics.Color;
import android.graphics.ImageDecoder;
import android.graphics.Typeface;
import android.net.Uri;
import android.os.Bundle;
import android.os.Handler;
import android.os.Looper;
import android.text.InputFilter;
import android.text.InputType;
import android.util.Base64;
import android.view.Gravity;
import android.view.View;
import android.view.WindowManager;
import android.widget.Button;
import android.widget.EditText;
import android.widget.ImageView;
import android.widget.LinearLayout;
import android.widget.ProgressBar;
import android.widget.ScrollView;
import android.widget.TextView;

import java.io.File;
import java.io.FileOutputStream;
import java.io.IOException;
import java.io.InputStream;
import java.io.RandomAccessFile;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.nio.charset.StandardCharsets;
import java.util.ArrayList;
import java.util.List;
import java.util.Locale;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;

public final class MainActivity extends Activity {
    private static final int PICK_IMAGE = 1001;
    private static final int PATCH_SIZE = 14;
    private static final int SIZE_DIVISOR = PATCH_SIZE * 4;
    private static final int SCALE_RESOLUTION = 448;
    private static final int MAX_SLICE_NUMS = 9;
    private static final int TILE_FILE_MAGIC = 0x4d435034;
    private static final int TILE_FILE_VERSION = 1;
    private static final int LETTERBOX_WIDTH = 392;
    private static final int LETTERBOX_HEIGHT = 504;
    private static final String ROOT = BuildConfig.RUNTIME_ROOT;
    private static final File BRIDGE_INPUT = new File(ROOT, "app-input-fp32.bin");
    private static final File BRIDGE_PROMPT = new File(ROOT, "app-prompt.txt");
    private static final File BRIDGE_REQUEST = new File(ROOT, "app-request");
    private static final File BRIDGE_STOP = new File(ROOT, "app-stop");
    private static final File BRIDGE_LOG = new File(ROOT, "app-run.log");
    private static final File SERVICE_START = new File(ROOT, "app-service-start");
    private static final File SERVICE_READY = new File(ROOT, "app-service-ready");
    private static final File RUNTIME_VERSION = new File(ROOT, ".runtime-version");
    private static final String[] COMMON_REQUIRED_FILES = {
            "libMNN.so",
            "libneuron_runtime.so.9.3.1",
            "libneuron_adapter.so.9.3.1",
            "config.json",
            "llm_config.json",
            "llm.mnn",
            "llm.mnn.weight",
            "tokenizer.mtok"
    };

    private final Handler main = new Handler(Looper.getMainLooper());
    private final ExecutorService worker = Executors.newSingleThreadExecutor();

    private TextView statusView;
    private TextView ttftView;
    private TextView outputView;
    private TextView metricsView;
    private ImageView imageView;
    private ProgressBar progressBar;
    private Button chooseButton;
    private Button runButton;
    private EditText promptView;
    private File imageTensor;
    private volatile boolean inferenceRunning;
    private boolean filesReady;

    @Override
    protected void onCreate(Bundle state) {
        super.onCreate(state);
        setContentView(buildContent());
        worker.execute(this::prepareRuntime);
    }

    private View buildContent() {
        float density = getResources().getDisplayMetrics().density;
        int padding = Math.round(18 * density);

        LinearLayout content = new LinearLayout(this);
        content.setOrientation(LinearLayout.VERTICAL);
        content.setPadding(padding, padding, padding, padding);

        TextView title = new TextView(this);
        title.setText(BuildConfig.APP_TITLE);
        title.setTextSize(24);
        title.setTextColor(0xff111111);
        title.setTypeface(Typeface.DEFAULT, Typeface.BOLD);
        content.addView(title);

        TextView route = new TextView(this);
        route.setText("NeuroPilot FP16 vision + MNN Q4 language model\n"
                + BuildConfig.IMAGE_MODE_LABEL);
        route.setTextSize(14);
        route.setTextColor(0xff555555);
        route.setPadding(0, padding / 4, 0, padding / 2);
        content.addView(route);

        statusView = new TextView(this);
        statusView.setText("Preparing packaged runtime...");
        statusView.setTextSize(15);
        statusView.setTextColor(0xff333333);
        statusView.setPadding(0, 0, 0, padding / 2);
        content.addView(statusView);

        imageView = new ImageView(this);
        imageView.setAdjustViewBounds(true);
        imageView.setScaleType(ImageView.ScaleType.CENTER_INSIDE);
        imageView.setBackgroundColor(0xffeeeeee);
        content.addView(imageView, new LinearLayout.LayoutParams(
                LinearLayout.LayoutParams.MATCH_PARENT, Math.round(300 * density)));

        TextView promptLabel = new TextView(this);
        promptLabel.setText("Prompt");
        promptLabel.setTextSize(14);
        promptLabel.setTextColor(0xff333333);
        promptLabel.setTypeface(Typeface.DEFAULT, Typeface.BOLD);
        promptLabel.setPadding(0, padding, 0, padding / 4);
        content.addView(promptLabel);

        promptView = new EditText(this);
        promptView.setText(BuildConfig.FIXED_PROMPT);
        promptView.setTextSize(14);
        promptView.setTextColor(0xff444444);
        promptView.setMinLines(2);
        promptView.setMaxLines(4);
        promptView.setInputType(InputType.TYPE_CLASS_TEXT
                | InputType.TYPE_TEXT_FLAG_MULTI_LINE
                | InputType.TYPE_TEXT_FLAG_CAP_SENTENCES);
        promptView.setFilters(new InputFilter[]{new InputFilter.LengthFilter(2048)});
        content.addView(promptView);

        LinearLayout actions = new LinearLayout(this);
        actions.setOrientation(LinearLayout.HORIZONTAL);
        actions.setPadding(0, padding / 2, 0, 0);

        chooseButton = new Button(this);
        chooseButton.setText("Choose image");
        chooseButton.setEnabled(false);
        chooseButton.setOnClickListener(view -> chooseImage());
        actions.addView(chooseButton, new LinearLayout.LayoutParams(
                0, LinearLayout.LayoutParams.WRAP_CONTENT, 1));

        runButton = new Button(this);
        runButton.setText("Run");
        runButton.setEnabled(false);
        runButton.setOnClickListener(view -> {
            if (inferenceRunning) {
                stopInference();
            } else {
                runInference();
            }
        });
        actions.addView(runButton, new LinearLayout.LayoutParams(
                0, LinearLayout.LayoutParams.WRAP_CONTENT, 1));
        content.addView(actions);

        ttftView = new TextView(this);
        ttftView.setText("TTFT  --");
        ttftView.setGravity(Gravity.CENTER);
        ttftView.setTextSize(28);
        ttftView.setTextColor(0xff087443);
        ttftView.setTypeface(Typeface.MONOSPACE, Typeface.BOLD);
        ttftView.setPadding(0, padding, 0, padding / 2);
        content.addView(ttftView);

        progressBar = new ProgressBar(this, null, android.R.attr.progressBarStyleHorizontal);
        progressBar.setMax(1024);
        progressBar.setVisibility(View.INVISIBLE);
        content.addView(progressBar, new LinearLayout.LayoutParams(
                LinearLayout.LayoutParams.MATCH_PARENT,
                LinearLayout.LayoutParams.WRAP_CONTENT));

        outputView = new TextView(this);
        outputView.setTextSize(16);
        outputView.setTextColor(0xff151515);
        outputView.setTextIsSelectable(true);
        outputView.setPadding(0, padding, 0, padding / 2);
        content.addView(outputView);

        metricsView = new TextView(this);
        metricsView.setTextSize(13);
        metricsView.setTextColor(0xff555555);
        metricsView.setTypeface(Typeface.MONOSPACE);
        metricsView.setTextIsSelectable(true);
        content.addView(metricsView);

        ScrollView scroll = new ScrollView(this);
        scroll.addView(content);
        return scroll;
    }

    private void prepareRuntime() {
        List<String> missing = new ArrayList<>();
        try {
            installPackagedRuntime();
        } catch (Exception error) {
            missing.add("Runtime extraction failed: " + error.getMessage());
        }
        for (String relative : requiredFiles()) {
            if (!new File(ROOT, relative).isFile()) {
                missing.add(relative);
            }
        }
        filesReady = missing.isEmpty();
        if (filesReady) {
            main.post(() -> statusView.setText("Loading NPU and MNN models..."));
            try {
                SERVICE_READY.delete();
                try (FileOutputStream ignored = new FileOutputStream(SERVICE_START)) {
                }
                long deadline = System.currentTimeMillis() + 300_000;
                while (!SERVICE_READY.isFile()) {
                    if (System.currentTimeMillis() > deadline) {
                        throw new IllegalStateException(
                                "Runtime service did not start. Run the included setup script.");
                    }
                    Thread.sleep(50);
                }
            } catch (Exception error) {
                filesReady = false;
                missing.add(error.getMessage());
            }
        }
        main.post(() -> {
            if (!filesReady) {
                statusView.setText("Runtime is not ready");
                metricsView.setText(String.join("\n", missing));
            } else {
                statusView.setText("Models loaded. Choose an image.");
                chooseButton.setEnabled(true);
            }
        });
    }

    private static List<String> requiredFiles() {
        List<String> files = new ArrayList<>();
        files.add(BuildConfig.RUNNER_NAME);
        files.add(BuildConfig.BRIDGE_NAME);
        for (String relative : COMMON_REQUIRED_FILES) {
            files.add(relative);
        }
        for (String relative : BuildConfig.VISION_MODEL_FILES.split(";")) {
            if (!relative.isEmpty()) {
                files.add(relative);
            }
        }
        return files;
    }

    private void installPackagedRuntime() throws Exception {
        boolean replaceAll = !runtimeVersionMatches();
        List<String> packagedFiles = new ArrayList<>();
        collectPackagedFiles("runtime", "", packagedFiles);
        for (int index = 0; index < packagedFiles.size(); index++) {
            String relative = packagedFiles.get(index);
            String assetPath = "runtime/" + relative;
            File destination = new File(ROOT, relative);
            long packagedSize = packagedAssetSize(assetPath);
            if (!replaceAll && destination.isFile() && packagedSize >= 0
                    && destination.length() == packagedSize) {
                continue;
            }
            int current = index + 1;
            int total = packagedFiles.size();
            main.post(() -> statusView.setText(String.format(Locale.US,
                    "Installing runtime %d/%d: %s", current, total, relative)));
            copyPackagedFile(assetPath, destination);
        }
        writeRuntimeVersion();
    }

    private boolean runtimeVersionMatches() {
        if (!RUNTIME_VERSION.isFile()) {
            return false;
        }
        try (RandomAccessFile input = new RandomAccessFile(RUNTIME_VERSION, "r")) {
            byte[] value = new byte[(int) input.length()];
            input.readFully(value);
            return BuildConfig.RUNTIME_ASSET_VERSION.equals(
                    new String(value, StandardCharsets.UTF_8));
        } catch (IOException ignored) {
            return false;
        }
    }

    private void writeRuntimeVersion() throws IOException {
        try (FileOutputStream output = new FileOutputStream(RUNTIME_VERSION)) {
            output.write(BuildConfig.RUNTIME_ASSET_VERSION.getBytes(StandardCharsets.UTF_8));
            output.getFD().sync();
        }
        RUNTIME_VERSION.setReadable(true, false);
        RUNTIME_VERSION.setWritable(true, false);
    }

    private void collectPackagedFiles(
            String assetPath, String relativePath, List<String> output) throws IOException {
        String[] children = getAssets().list(assetPath);
        if (children == null || children.length == 0) {
            if (!relativePath.isEmpty()) {
                output.add(relativePath);
            }
            return;
        }
        for (String child : children) {
            String childAsset = assetPath + "/" + child;
            String childRelative = relativePath.isEmpty()
                    ? child : relativePath + "/" + child;
            collectPackagedFiles(childAsset, childRelative, output);
        }
    }

    private long packagedAssetSize(String assetPath) {
        try (AssetFileDescriptor descriptor = getAssets().openFd(assetPath)) {
            return descriptor.getLength();
        } catch (IOException ignored) {
            try (InputStream input = getAssets().open(assetPath)) {
                return input.available();
            } catch (IOException unavailable) {
                return -1;
            }
        }
    }

    private void copyPackagedFile(String assetPath, File destination) throws Exception {
        File parent = destination.getParentFile();
        if (parent == null || (!parent.isDirectory() && !parent.mkdirs())) {
            throw new IOException("Cannot create " + parent);
        }
        File temporary = new File(destination.getPath() + ".part");
        try (InputStream input = getAssets().open(assetPath);
                FileOutputStream output = new FileOutputStream(temporary)) {
            byte[] buffer = new byte[8 * 1024 * 1024];
            int count;
            while ((count = input.read(buffer)) != -1) {
                output.write(buffer, 0, count);
            }
            output.getFD().sync();
        }
        if (destination.exists() && !destination.delete()) {
            throw new IOException("Cannot replace " + destination);
        }
        if (!temporary.renameTo(destination)) {
            throw new IOException("Cannot publish " + destination);
        }
        destination.setReadable(true, false);
        destination.setWritable(true, false);
        if (destination.getName().equals(BuildConfig.RUNNER_NAME)
                || destination.getName().endsWith(".sh")) {
            destination.setExecutable(true, false);
        }
    }

    private void chooseImage() {
        Intent intent = new Intent(Intent.ACTION_OPEN_DOCUMENT);
        intent.addCategory(Intent.CATEGORY_OPENABLE);
        intent.setType("image/*");
        startActivityForResult(intent, PICK_IMAGE);
    }

    @Override
    protected void onActivityResult(int requestCode, int resultCode, Intent data) {
        super.onActivityResult(requestCode, resultCode, data);
        if (requestCode != PICK_IMAGE || resultCode != RESULT_OK || data == null
                || data.getData() == null) {
            return;
        }
        chooseButton.setEnabled(false);
        runButton.setEnabled(false);
        statusView.setText("Preparing the image tensor...");
        worker.execute(() -> prepareImage(data.getData()));
    }

    private void prepareImage(Uri uri) {
        try {
            ImageDecoder.Source imageSource = ImageDecoder.createSource(getContentResolver(), uri);
            Bitmap source = ImageDecoder.decodeBitmap(imageSource,
                    (decoder, info, src) -> decoder.setAllocator(ImageDecoder.ALLOCATOR_SOFTWARE));
            SliceGrid grid = null;
            List<Bitmap> tiles;
            String selectedMode;
            if ("letterbox".equals(BuildConfig.IMAGE_PREPROCESSOR)) {
                tiles = new ArrayList<>();
                tiles.add(buildLetterboxImage(source));
                selectedMode = "Letterbox, 63 visual tokens";
            } else {
                grid = chooseSliceGrid(source.getHeight(), source.getWidth());
                tiles = buildOfficialTiles(source, grid);
                if ("adaptive".equals(BuildConfig.IMAGE_PREPROCESSOR)
                        && !allVisionProfilesAvailable(tiles)) {
                    grid = null;
                    tiles = new ArrayList<>();
                    tiles.add(buildLetterboxImage(source));
                    selectedMode = "Automatic letterbox fallback, 63 visual tokens";
                } else {
                    selectedMode = "Official dynamic multi-tile";
                }
            }
            imageTensor = BRIDGE_INPUT;
            writeDynamicTilesFp32(tiles, grid, imageTensor);
            Bitmap globalImage = tiles.get(0);
            int tileCount = tiles.size();
            String imageMode = selectedMode;
            main.post(() -> {
                imageView.setImageBitmap(globalImage);
                statusView.setText(String.format(Locale.US,
                        "Image ready: %d visual tile%s (%s)", tileCount,
                        tileCount == 1 ? "" : "s", imageMode));
                chooseButton.setEnabled(true);
                runButton.setEnabled(true);
            });
        } catch (Exception error) {
            main.post(() -> {
                statusView.setText("Image preparation failed: " + error.getMessage());
                chooseButton.setEnabled(true);
            });
        }
    }

    private static boolean allVisionProfilesAvailable(List<Bitmap> tiles) {
        String packagedProfiles = ";" + BuildConfig.VISION_MODEL_FILES + ";";
        for (Bitmap tile : tiles) {
            String profile = "models/visual_" + tile.getHeight() + "x"
                    + tile.getWidth() + "_fp16_mt6899.dla";
            if (!packagedProfiles.contains(";" + profile + ";")) {
                return false;
            }
        }
        return true;
    }

    private static Bitmap buildLetterboxImage(Bitmap source) {
        double scale = Math.min(
                (double) LETTERBOX_WIDTH / source.getWidth(),
                (double) LETTERBOX_HEIGHT / source.getHeight());
        int width = Math.max(1, Math.min(LETTERBOX_WIDTH,
                (int) Math.round(source.getWidth() * scale)));
        int height = Math.max(1, Math.min(LETTERBOX_HEIGHT,
                (int) Math.round(source.getHeight() * scale)));
        Bitmap resized = bicubicResize(source, width, height);
        Bitmap canvasBitmap = Bitmap.createBitmap(
                LETTERBOX_WIDTH, LETTERBOX_HEIGHT, Bitmap.Config.ARGB_8888);
        canvasBitmap.eraseColor(Color.rgb(128, 128, 128));
        Canvas canvas = new Canvas(canvasBitmap);
        canvas.drawBitmap(
                resized,
                (float) ((LETTERBOX_WIDTH - width) / 2),
                (float) ((LETTERBOX_HEIGHT - height) / 2),
                null);
        return canvasBitmap;
    }

    private static final class SliceGrid {
        final int rows;
        final int columns;

        SliceGrid(int rows, int columns) {
            this.rows = rows;
            this.columns = columns;
        }
    }

    private static int ensureDivide(int length, int divisor) {
        return Math.max(Math.round((float) length / divisor) * divisor, divisor);
    }

    private static int[] findBestResize(
            int height, int width, boolean allowUpscale) {
        if ((long) height * width > (long) SCALE_RESOLUTION * SCALE_RESOLUTION
                || allowUpscale) {
            double aspectRatio = (double) width / height;
            height = (int) (SCALE_RESOLUTION / Math.sqrt(aspectRatio));
            width = (int) (height * aspectRatio);
        }
        return new int[] {
                ensureDivide(height, SIZE_DIVISOR),
                ensureDivide(width, SIZE_DIVISOR)
        };
    }

    private static SliceGrid chooseSliceGrid(int height, int width) {
        double ratio = (double) width * height
                / ((double) SCALE_RESOLUTION * SCALE_RESOLUTION);
        int multiple = Math.min((int) Math.ceil(ratio), MAX_SLICE_NUMS);
        if (multiple <= 1) {
            return null;
        }

        double logRatio = Math.log((double) width / height);
        double minimumError = Double.POSITIVE_INFINITY;
        SliceGrid best = new SliceGrid(1, 1);
        for (int slices = multiple - 1; slices <= multiple + 1; slices++) {
            if (slices <= 1 || slices > MAX_SLICE_NUMS) {
                continue;
            }
            for (int candidateRows = 1; candidateRows <= slices; candidateRows++) {
                if (slices % candidateRows != 0) {
                    continue;
                }
                int candidateColumns = slices / candidateRows;
                double error = Math.abs(logRatio
                        - Math.log((double) candidateRows / candidateColumns));
                if (error < minimumError) {
                    minimumError = error;
                    best = new SliceGrid(candidateColumns, candidateRows);
                }
            }
        }
        return best;
    }

    private static int[] getRefineSize(
            int height, int width, SliceGrid grid) {
        int refineWidth = ensureDivide(width, grid.columns);
        int refineHeight = ensureDivide(height, grid.rows);
        int[] tileSize = findBestResize(
                refineHeight / grid.rows,
                refineWidth / grid.columns,
                true);
        return new int[] {
                tileSize[0] * grid.rows,
                tileSize[1] * grid.columns
        };
    }

    private static List<Bitmap> buildOfficialTiles(Bitmap source, SliceGrid grid) {
        List<Bitmap> tiles = new ArrayList<>();
        int[] globalSize = findBestResize(
                source.getHeight(), source.getWidth(), grid == null);
        tiles.add(bicubicResize(source, globalSize[1], globalSize[0]));
        if (grid == null) {
            return tiles;
        }

        int[] refineSize = getRefineSize(
                source.getHeight(), source.getWidth(), grid);
        Bitmap refined = bicubicResize(source, refineSize[1], refineSize[0]);
        int tileWidth = refineSize[1] / grid.columns;
        int tileHeight = refineSize[0] / grid.rows;
        for (int row = 0; row < grid.rows; row++) {
            for (int column = 0; column < grid.columns; column++) {
                tiles.add(Bitmap.createBitmap(
                        refined,
                        column * tileWidth,
                        row * tileHeight,
                        tileWidth,
                        tileHeight));
            }
        }
        return tiles;
    }

    private static Bitmap bicubicResize(Bitmap source, int width, int height) {
        int sourceWidth = source.getWidth();
        int sourceHeight = source.getHeight();
        int[] sourcePixels = new int[sourceWidth * sourceHeight];
        int[] targetPixels = new int[width * height];
        source.getPixels(sourcePixels, 0, sourceWidth, 0, 0, sourceWidth, sourceHeight);
        double scaleX = (double) sourceWidth / width;
        double scaleY = (double) sourceHeight / height;

        for (int y = 0; y < height; y++) {
            double sourceY = (y + 0.5) * scaleY - 0.5;
            int baseY = (int) Math.floor(sourceY);
            for (int x = 0; x < width; x++) {
                double sourceX = (x + 0.5) * scaleX - 0.5;
                int baseX = (int) Math.floor(sourceX);
                double red = 0.0;
                double green = 0.0;
                double blue = 0.0;
                double weightSum = 0.0;
                for (int ky = -1; ky <= 2; ky++) {
                    int sampleY = clamp(baseY + ky, 0, sourceHeight - 1);
                    double wy = cubicWeight(sourceY - (baseY + ky));
                    for (int kx = -1; kx <= 2; kx++) {
                        int sampleX = clamp(baseX + kx, 0, sourceWidth - 1);
                        double weight = wy * cubicWeight(sourceX - (baseX + kx));
                        int pixel = sourcePixels[sampleY * sourceWidth + sampleX];
                        red += ((pixel >> 16) & 0xff) * weight;
                        green += ((pixel >> 8) & 0xff) * weight;
                        blue += (pixel & 0xff) * weight;
                        weightSum += weight;
                    }
                }
                int r = clamp((int) Math.round(red / weightSum), 0, 255);
                int g = clamp((int) Math.round(green / weightSum), 0, 255);
                int b = clamp((int) Math.round(blue / weightSum), 0, 255);
                targetPixels[y * width + x] = 0xff000000 | (r << 16) | (g << 8) | b;
            }
        }
        return Bitmap.createBitmap(targetPixels, width, height, Bitmap.Config.ARGB_8888);
    }

    private static double cubicWeight(double value) {
        double x = Math.abs(value);
        if (x <= 1.0) {
            return 1.5 * x * x * x - 2.5 * x * x + 1.0;
        }
        if (x < 2.0) {
            return -0.5 * x * x * x + 2.5 * x * x - 4.0 * x + 2.0;
        }
        return 0.0;
    }

    private static int clamp(int value, int minimum, int maximum) {
        return Math.max(minimum, Math.min(maximum, value));
    }

    private static void writeDynamicTilesFp32(
            List<Bitmap> tiles, SliceGrid grid, File output) throws Exception {
        File temporary = new File(output.getPath() + ".part");
        try (FileOutputStream stream = new FileOutputStream(temporary)) {
            ByteBuffer header = ByteBuffer.allocate(5 * 4).order(ByteOrder.LITTLE_ENDIAN);
            header.putInt(TILE_FILE_MAGIC);
            header.putInt(TILE_FILE_VERSION);
            header.putInt(grid == null ? 0 : grid.rows);
            header.putInt(grid == null ? 0 : grid.columns);
            header.putInt(tiles.size());
            stream.write(header.array());

            int[] shifts = {16, 8, 0};
            for (Bitmap bitmap : tiles) {
                int width = bitmap.getWidth();
                int height = bitmap.getHeight();
                int[] pixels = new int[width * height];
                bitmap.getPixels(pixels, 0, width, 0, 0, width, height);
                ByteBuffer tile = ByteBuffer.allocate(2 * 4 + 3 * width * height * 4)
                        .order(ByteOrder.LITTLE_ENDIAN);
                tile.putInt(height);
                tile.putInt(width);
                for (int shift : shifts) {
                    for (int pixel : pixels) {
                        int channel = (pixel >> shift) & 0xff;
                        tile.putFloat(channel / 127.5f - 1.0f);
                    }
                }
                stream.write(tile.array());
            }
            stream.getFD().sync();
        }
        if (output.exists() && !output.delete()) {
            throw new IOException("Cannot replace image tensor");
        }
        if (!temporary.renameTo(output)) {
            throw new IOException("Cannot publish image tensor");
        }
    }

    private void runInference() {
        String prompt = promptView.getText().toString().trim();
        if (prompt.isEmpty()) {
            prompt = BuildConfig.FIXED_PROMPT;
            promptView.setText(prompt);
        }
        String requestPrompt = prompt;
        inferenceRunning = true;
        runButton.setText("Stop");
        chooseButton.setEnabled(false);
        promptView.setEnabled(false);
        ttftView.setText("TTFT  Running...");
        outputView.setText("");
        metricsView.setText("Prompt: " + requestPrompt + "\n");
        progressBar.setProgress(0);
        progressBar.setVisibility(View.VISIBLE);
        statusView.setText("NPU vision and MNN prefill running");
        getWindow().addFlags(WindowManager.LayoutParams.FLAG_KEEP_SCREEN_ON);

        worker.execute(() -> {
            try {
                BRIDGE_STOP.delete();
                writePrompt(requestPrompt);
                long requestedAt = System.currentTimeMillis();
                try (FileOutputStream ignored = new FileOutputStream(BRIDGE_REQUEST)) {
                }
                followBridgeLog(requestedAt);
            } catch (Exception error) {
                main.post(() -> finish("Inference failed: " + error.getMessage()));
            }
        });
    }

    private static void writePrompt(String prompt) throws IOException {
        File temporary = new File(BRIDGE_PROMPT.getPath() + ".part");
        try (FileOutputStream output = new FileOutputStream(temporary)) {
            output.write(prompt.getBytes(StandardCharsets.UTF_8));
            output.getFD().sync();
        }
        if (BRIDGE_PROMPT.exists() && !BRIDGE_PROMPT.delete()) {
            throw new IOException("Cannot replace prompt file");
        }
        if (!temporary.renameTo(BRIDGE_PROMPT)) {
            throw new IOException("Cannot publish prompt file");
        }
    }

    private void followBridgeLog(long requestedAt) throws Exception {
        long deadline = System.currentTimeMillis() + 15_000;
        while (!BRIDGE_LOG.isFile() || BRIDGE_LOG.lastModified() < requestedAt) {
            if (System.currentTimeMillis() > deadline) {
                throw new IllegalStateException("Runtime bridge is not running");
            }
            Thread.sleep(50);
        }

        try (RandomAccessFile log = new RandomAccessFile(BRIDGE_LOG, "r")) {
            while (inferenceRunning && !Thread.currentThread().isInterrupted()) {
                String rawLine = log.readLine();
                if (rawLine == null) {
                    Thread.sleep(50);
                    continue;
                }
                String line = new String(rawLine.getBytes(StandardCharsets.ISO_8859_1),
                        StandardCharsets.UTF_8);
                handleLine(line);
                if (line.startsWith("RESULT request_complete=") || line.startsWith("ERROR ")) {
                    main.post(() -> finish(line.startsWith("ERROR ")
                            ? "Inference failed" : "Inference complete"));
                    return;
                }
            }
        }
    }

    private void handleLine(String line) {
        if (line.startsWith("FIRST_TOKEN ")) {
            String ttft = field(line, "ttft_ms");
            main.post(() -> {
                ttftView.setText(String.format(Locale.US, "TTFT  %s ms", ttft));
                statusView.setText("First token ready; decoding continues");
                metricsView.append("TTFT: " + ttft + " ms\n");
            });
        } else if (line.startsWith("TOKEN ")) {
            int index = parseInt(field(line, "index"));
            String encoded = field(line, "text_b64");
            String piece;
            try {
                piece = new String(Base64.decode(encoded, Base64.DEFAULT), StandardCharsets.UTF_8);
            } catch (IllegalArgumentException error) {
                piece = "<?>";
            }
            String decodedPiece = piece;
            main.post(() -> {
                outputView.append(decodedPiece);
                progressBar.setProgress(Math.max(0, index), true);
                statusView.setText("Decoding: " + index + " tokens");
            });
        } else if (line.startsWith("RESULT ") || line.startsWith("ERROR ")) {
            main.post(() -> metricsView.append(line + "\n"));
        }
    }

    private static int parseInt(String value) {
        try {
            return Integer.parseInt(value);
        } catch (NumberFormatException ignored) {
            return -1;
        }
    }

    private static String field(String line, String name) {
        String prefix = name + "=";
        for (String part : line.split(" ")) {
            if (part.startsWith(prefix)) {
                return part.substring(prefix.length());
            }
        }
        return "?";
    }

    private void stopInference() {
        try (FileOutputStream ignored = new FileOutputStream(BRIDGE_STOP)) {
        } catch (Exception ignored) {
        }
        finish("Inference stopped");
    }

    private void finish(String status) {
        inferenceRunning = false;
        statusView.setText(status);
        runButton.setText("Run");
        runButton.setEnabled(imageTensor != null);
        chooseButton.setEnabled(filesReady);
        promptView.setEnabled(true);
        getWindow().clearFlags(WindowManager.LayoutParams.FLAG_KEEP_SCREEN_ON);
    }

    @Override
    protected void onDestroy() {
        if (inferenceRunning) {
            try (FileOutputStream ignored = new FileOutputStream(BRIDGE_STOP)) {
            } catch (Exception ignored) {
            }
        }
        inferenceRunning = false;
        worker.shutdownNow();
        super.onDestroy();
    }
}
