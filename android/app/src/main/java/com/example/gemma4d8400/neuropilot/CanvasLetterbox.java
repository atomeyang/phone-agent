package com.example.gemma4d8400.neuropilot;

import android.graphics.Bitmap;
import android.graphics.Canvas;
import android.graphics.Color;

import java.io.File;
import java.io.FileOutputStream;
import java.io.IOException;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;

/**
 * Turns a gallery bitmap into the canvas file the NPU vision graph expects.
 *
 * <p>The DLA is compiled for a fixed patch grid, so the geometry is a build
 * time constant: the image is aspect-preserving resized into the top-left of a
 * {@code gridColumns x gridRows} patch canvas and the padding stays black (the
 * runner masks those patches out of the vision attention).  The file layout is
 * the one {@code gemma4_hybrid_runner} documents: magic, version, valid patch
 * rows/columns, tile count, then one tile of planar RGB float32 in [0, 1].
 */
public final class CanvasLetterbox {

    private static final int PATCH_SIZE = 16;
    private static final int POOL_SIZE = 3;
    private static final int SIDE_MULTIPLE = PATCH_SIZE * POOL_SIZE;
    private static final int TILE_FILE_MAGIC = 0x4d435034;
    private static final int TILE_FILE_VERSION = 1;

    private final int gridColumns;
    private final int gridRows;

    public CanvasLetterbox(int gridColumns, int gridRows) {
        this.gridColumns = gridColumns;
        this.gridRows = gridRows;
    }

    public static final class Result {
        public final Bitmap preview;
        public final int patchRows;
        public final int patchColumns;
        public final int softTokens;

        Result(Bitmap preview, int patchRows, int patchColumns) {
            this.preview = preview;
            this.patchRows = patchRows;
            this.patchColumns = patchColumns;
            this.softTokens = (patchRows / POOL_SIZE) * (patchColumns / POOL_SIZE);
        }
    }

    /**
     * Official Gemma 4 resize rule: keep the aspect ratio, stay inside the patch
     * budget and floor both sides to a multiple of pooling kernel times patch
     * size (48 px), matching
     * {@code transformers.models.gemma4.image_processing_gemma4}.
     */
    public Grid buildGrid(int height, int width) {
        final double targetPixels = (double) gridColumns * gridRows * PATCH_SIZE * PATCH_SIZE;
        final double factor = Math.sqrt(targetPixels / ((double) height * width));
        int targetHeight = (int) (Math.floor(factor * height / SIDE_MULTIPLE) * SIDE_MULTIPLE);
        int targetWidth = (int) (Math.floor(factor * width / SIDE_MULTIPLE) * SIDE_MULTIPLE);
        if (targetHeight == 0 && targetWidth == 0) {
            throw new IllegalArgumentException("Image is too small to resize");
        }
        targetHeight = Math.max(targetHeight, SIDE_MULTIPLE);
        targetWidth = Math.max(targetWidth, SIDE_MULTIPLE);
        targetHeight = Math.min(targetHeight, gridRows * PATCH_SIZE);
        targetWidth = Math.min(targetWidth, gridColumns * PATCH_SIZE);
        targetHeight = Math.max(SIDE_MULTIPLE, (targetHeight / SIDE_MULTIPLE) * SIDE_MULTIPLE);
        targetWidth = Math.max(SIDE_MULTIPLE, (targetWidth / SIDE_MULTIPLE) * SIDE_MULTIPLE);
        return new Grid(targetHeight / PATCH_SIZE, targetWidth / PATCH_SIZE);
    }

    public static final class Grid {
        public final int rows;
        public final int columns;

        Grid(int rows, int columns) {
            this.rows = rows;
            this.columns = columns;
        }
    }

    /** Writes the canvas tensor and returns a preview of what the NPU sees. */
    public Result prepare(Bitmap source, File output) throws IOException {
        final Grid grid = buildGrid(source.getHeight(), source.getWidth());
        final Bitmap canvasBitmap = buildCanvasImage(source, grid);
        writeTilesFp32(canvasBitmap, grid, output);
        return new Result(canvasBitmap, grid.rows, grid.columns);
    }

    private Bitmap buildCanvasImage(Bitmap source, Grid grid) {
        final int width = grid.columns * PATCH_SIZE;
        final int height = grid.rows * PATCH_SIZE;
        final Bitmap resized = bicubicResize(source, width, height);
        final Bitmap canvasBitmap = Bitmap.createBitmap(
                gridColumns * PATCH_SIZE, gridRows * PATCH_SIZE, Bitmap.Config.ARGB_8888);
        // Padding patches are zero; the runner masks them out of the attention,
        // so the value only has to be constant.
        canvasBitmap.eraseColor(Color.rgb(0, 0, 0));
        final Canvas canvas = new Canvas(canvasBitmap);
        canvas.drawBitmap(resized, 0.0f, 0.0f, null);
        return canvasBitmap;
    }

    private void writeTilesFp32(Bitmap bitmap, Grid grid, File output) throws IOException {
        final int width = bitmap.getWidth();
        final int height = bitmap.getHeight();
        final int[] pixels = new int[width * height];
        bitmap.getPixels(pixels, 0, width, 0, 0, width, height);

        final File temporary = new File(output.getPath() + ".part");
        try (FileOutputStream stream = new FileOutputStream(temporary)) {
            final ByteBuffer header = ByteBuffer.allocate(5 * 4).order(ByteOrder.LITTLE_ENDIAN);
            header.putInt(TILE_FILE_MAGIC);
            header.putInt(TILE_FILE_VERSION);
            header.putInt(grid.rows);
            header.putInt(grid.columns);
            header.putInt(1);
            stream.write(header.array());

            final ByteBuffer tile = ByteBuffer.allocate(2 * 4 + 3 * width * height * 4)
                    .order(ByteOrder.LITTLE_ENDIAN);
            tile.putInt(height);
            tile.putInt(width);
            final int[] shifts = {16, 8, 0};
            for (final int shift : shifts) {
                for (final int pixel : pixels) {
                    // The Gemma 4 vision graph consumes pixels in [0, 1].
                    tile.putFloat(((pixel >> shift) & 0xff) / 255.0f);
                }
            }
            stream.write(tile.array());
            stream.getFD().sync();
        }
        if (output.exists() && !output.delete()) {
            throw new IOException("Cannot replace the image tensor");
        }
        if (!temporary.renameTo(output)) {
            throw new IOException("Cannot publish the image tensor");
        }
    }

    private static Bitmap bicubicResize(Bitmap source, int width, int height) {
        final int sourceWidth = source.getWidth();
        final int sourceHeight = source.getHeight();
        final int[] sourcePixels = new int[sourceWidth * sourceHeight];
        final int[] targetPixels = new int[width * height];
        source.getPixels(sourcePixels, 0, sourceWidth, 0, 0, sourceWidth, sourceHeight);
        final double scaleX = (double) sourceWidth / width;
        final double scaleY = (double) sourceHeight / height;

        for (int y = 0; y < height; y++) {
            final double sourceY = (y + 0.5) * scaleY - 0.5;
            final int baseY = (int) Math.floor(sourceY);
            for (int x = 0; x < width; x++) {
                final double sourceX = (x + 0.5) * scaleX - 0.5;
                final int baseX = (int) Math.floor(sourceX);
                double red = 0.0;
                double green = 0.0;
                double blue = 0.0;
                double weightSum = 0.0;
                for (int ky = -1; ky <= 2; ky++) {
                    final int sampleY = clamp(baseY + ky, 0, sourceHeight - 1);
                    final double wy = cubicWeight(sourceY - (baseY + ky));
                    for (int kx = -1; kx <= 2; kx++) {
                        final int sampleX = clamp(baseX + kx, 0, sourceWidth - 1);
                        final double weight = wy * cubicWeight(sourceX - (baseX + kx));
                        final int pixel = sourcePixels[sampleY * sourceWidth + sampleX];
                        red += ((pixel >> 16) & 0xff) * weight;
                        green += ((pixel >> 8) & 0xff) * weight;
                        blue += (pixel & 0xff) * weight;
                        weightSum += weight;
                    }
                }
                final int r = clamp((int) Math.round(red / weightSum), 0, 255);
                final int g = clamp((int) Math.round(green / weightSum), 0, 255);
                final int b = clamp((int) Math.round(blue / weightSum), 0, 255);
                targetPixels[y * width + x] = 0xff000000 | (r << 16) | (g << 8) | b;
            }
        }
        return Bitmap.createBitmap(targetPixels, width, height, Bitmap.Config.ARGB_8888);
    }

    private static double cubicWeight(double value) {
        final double x = Math.abs(value);
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
}
