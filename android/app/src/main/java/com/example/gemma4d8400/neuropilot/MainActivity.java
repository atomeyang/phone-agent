package com.example.gemma4d8400.neuropilot;

import android.app.Activity;
import android.app.AlertDialog;
import android.content.Intent;
import android.graphics.Bitmap;
import android.graphics.ImageDecoder;
import android.graphics.Typeface;
import android.graphics.drawable.GradientDrawable;
import android.net.Uri;
import android.os.Bundle;
import android.os.Handler;
import android.os.Looper;
import android.os.PowerManager;
import android.os.SystemClock;
import android.text.InputType;
import android.util.Log;
import android.util.TypedValue;
import android.view.Gravity;
import android.view.View;
import android.view.ViewGroup;
import android.view.WindowManager;
import android.widget.Button;
import android.widget.EditText;
import android.widget.FrameLayout;
import android.widget.ImageView;
import android.widget.LinearLayout;
import android.widget.ScrollView;
import android.widget.TextView;

import org.json.JSONArray;
import org.json.JSONObject;

import java.io.File;
import java.io.FileOutputStream;
import java.io.FileWriter;
import java.io.IOException;
import java.io.PrintWriter;
import java.io.RandomAccessFile;
import java.nio.charset.StandardCharsets;
import java.util.ArrayList;
import java.util.List;
import java.util.Locale;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;

/**
 * On-device agent UI.
 *
 * <p>The activity is a renderer and a control surface only: model inference,
 * the memory store, planning, tool execution and conversation history all live
 * in the native runner, which this app talks to through
 * {@link AgentClient}.  The tabs map to what the native side exposes - chat
 * with the per-turn plan/tool trace, the memory store, the session list and the
 * runtime/benchmark view.
 */
public final class MainActivity extends Activity {

    private static final String TAG = "Gemma4Agent";
    private static final int PICK_IMAGE = 1001;
    private static final int COLOR_BG = 0xffffffff;
    private static final int COLOR_PANEL = 0xfff7f8fa;
    private static final int COLOR_TEXT = 0xff14161a;
    private static final int COLOR_MUTED = 0xff5f6673;
    private static final int COLOR_ACCENT = 0xff1b6ef3;
    private static final int COLOR_WARN = 0xffb3541e;

    private static final File ROOT = new File(BuildConfig.RUNTIME_ROOT);
    private static final File BRIDGE_INPUT = new File(ROOT, "app-input-fp32.bin");
    private static final File BRIDGE_PROMPT = new File(ROOT, "app-prompt.txt");
    private static final File BRIDGE_REQUEST = new File(ROOT, "app-request");
    private static final File BRIDGE_STOP = new File(ROOT, "app-stop");
    private static final File BRIDGE_LOG = new File(ROOT, "app-run.log");
    private static final File SERVICE_START = new File(ROOT, "app-service-start");
    private static final File PLATFORM_CONFIG = new File(ROOT, ".platform");
    /**
     * Test hook: a root shell can write this file to drive the UI the way a
     * person would - the text lands in the input box, the image is attached
     * through the same code path the picker uses, and "send" presses the same
     * button.  It exists so the scenario suite can run against the real UI on a
     * device without a software keyboard for CJK text.
     */
    private static final File UI_COMMAND = new File(ROOT, "app-ui-command.json");

    private final Handler main = new Handler(Looper.getMainLooper());
    private final ExecutorService worker = Executors.newSingleThreadExecutor();
    private final CanvasLetterbox letterbox =
            new CanvasLetterbox(BuildConfig.GRID_COLUMNS, BuildConfig.GRID_ROWS);

    private AgentClient client;
    private ChatTranscriptView transcript;

    private TextView statusView;
    private TextView headerView;
    private LinearLayout panelHost;
    private LinearLayout chatPanel;
    private LinearLayout memoryPanel;
    private LinearLayout sessionPanel;
    private LinearLayout modelPanel;
    private LinearLayout chatList;
    private LinearLayout memoryList;
    private LinearLayout sessionList;
    private LinearLayout toolsList;
    private TextView runtimeView;
    private TextView perfView;
    private TextView logView;
    private ScrollView chatScroll;
    private EditText inputView;
    private Button sendButton;
    private Button attachButton;
    private ImageView previewView;
    private TextView attachLabel;
    private Button removeButton;
    private android.widget.Switch memoryToggle;
    private android.widget.Switch historyToggle;
    private final List<Button> tabButtons = new ArrayList<>();

    private File attachedCanvas;
    private final java.util.List<Button> quickButtons = new java.util.ArrayList<>();
    private String attachedNote;
    private boolean filesReady;
    private boolean inferenceRunning;
    private PowerManager.WakeLock turnWakeLock;
    private PowerManager.WakeLock benchmarkWakeLock;
    private PerfMonitor perfMonitor;
    private PrintWriter perfLogWriter;
    private long inferenceStartElapsed;
    private long firstTokenElapsed;
    private int decodeTokenTotal;
    private long firstDecodeSampleElapsed;
    private long lastTokenElapsed;
    private int lastTokenIndex = -1;
    private static volatile MainActivity instance;

    // --- lifecycle ----------------------------------------------------------

    @Override
    protected void onCreate(Bundle state) {
        super.onCreate(state);
        instance = this;
        client = new AgentClient(ROOT);
        setContentView(buildContent());
        showTab(0);
        worker.execute(this::prepareRuntime);
    }

    @Override
    protected void onDestroy() {
        if (turnRunning()) {
            client.cancelCurrentTurn();
        }
        if (client != null) {
            client.stopPolling();
        }
        worker.shutdownNow();
        releaseWakeLocks();
        closePerfLog();
        instance = null;
        super.onDestroy();
    }

    /** Called by {@link BenchmarkReceiver} to run the packaged benchmark. */
    static void triggerBenchmarkFromReceiver(android.content.Context context, boolean screenOff) {
        Log.i(TAG, "benchmark requested screenOff=" + screenOff);
        final PowerManager manager =
                (PowerManager) context.getSystemService(POWER_SERVICE);
        if (manager != null && instance != null) {
            final PowerManager.WakeLock lock =
                    manager.newWakeLock(PowerManager.PARTIAL_WAKE_LOCK, "Gemma4Agent:bench");
            lock.acquire(600_000L);
            instance.benchmarkWakeLock = lock;
        }
        if (instance != null && instance.filesReady && instance.attachedCanvas != null) {
            instance.main.post(() -> instance.runPerfScenario(!screenOff));
            return;
        }
        Log.w(TAG, "benchmark ignored: the runtime or the image is not ready");
    }

    // --- UI construction ----------------------------------------------------

    private View buildContent() {
        final LinearLayout root = new LinearLayout(this);
        root.setOrientation(LinearLayout.VERTICAL);
        root.setBackgroundColor(COLOR_BG);
        final int padding = dp(14);
        root.setPadding(padding, padding, padding, 0);

        headerView = text("Gemma 4 E2B 端侧 Agent", 20, COLOR_TEXT, true);
        root.addView(headerView);
        headerView = text("native inference · memory · planning · tools · history",
                12, COLOR_MUTED, false);
        root.addView(headerView);

        statusView = text("正在准备运行环境…", 12, COLOR_ACCENT, false);
        statusView.setPadding(0, dp(6), 0, dp(6));
        root.addView(statusView);

        final LinearLayout tabs = new LinearLayout(this);
        tabs.setOrientation(LinearLayout.HORIZONTAL);
        final String[] names = {"对话", "记忆", "会话", "模型"};
        for (int index = 0; index < names.length; index++) {
            final int tabIndex = index;
            final Button button = new Button(this);
            button.setText(names[index]);
            button.setTextSize(13);
            button.setAllCaps(false);
            button.setOnClickListener(view -> showTab(tabIndex));
            tabs.addView(button, new LinearLayout.LayoutParams(
                    0, ViewGroup.LayoutParams.WRAP_CONTENT, 1f));
            tabButtons.add(button);
        }
        root.addView(tabs);

        panelHost = new LinearLayout(this);
        panelHost.setOrientation(LinearLayout.VERTICAL);
        root.addView(panelHost, new LinearLayout.LayoutParams(
                ViewGroup.LayoutParams.MATCH_PARENT, 0, 1f));

        chatPanel = buildChatPanel();
        memoryPanel = buildListPanel(memoryList = new LinearLayout(this), "记忆库");
        sessionPanel = buildListPanel(sessionList = new LinearLayout(this), "会话");
        modelPanel = buildModelPanel();
        panelHost.addView(chatPanel);
        panelHost.addView(memoryPanel);
        panelHost.addView(sessionPanel);
        panelHost.addView(modelPanel);
        return root;
    }

    private LinearLayout buildListPanel(LinearLayout list, String title) {
        final LinearLayout panel = new LinearLayout(this);
        panel.setOrientation(LinearLayout.VERTICAL);
        panel.setVisibility(View.GONE);
        final TextView header = text(title, 14, COLOR_TEXT, true);
        header.setPadding(0, dp(6), 0, dp(6));
        panel.addView(header);
        final ScrollView scroll = new ScrollView(this);
        list.setOrientation(LinearLayout.VERTICAL);
        scroll.addView(list);
        panel.addView(scroll, new LinearLayout.LayoutParams(
                ViewGroup.LayoutParams.MATCH_PARENT, 0, 1f));
        return panel;
    }

    private LinearLayout buildChatPanel() {
        final LinearLayout panel = new LinearLayout(this);
        panel.setOrientation(LinearLayout.VERTICAL);

        chatList = new LinearLayout(this);
        chatList.setOrientation(LinearLayout.VERTICAL);
        chatScroll = new ScrollView(this);
        chatScroll.addView(chatList);
        panel.addView(chatScroll, new LinearLayout.LayoutParams(
                ViewGroup.LayoutParams.MATCH_PARENT, 0, 1f));
        transcript = new ChatTranscriptView(this, chatList, chatScroll);
        transcript.setListener(new ChatTranscriptView.Listener() {
            @Override
            public void onTurnFinished(boolean ok) {
                finishTurn(ok ? "回答完成" : "本轮结束");
            }

            @Override
            public void onStatus(String message) {
                status(message);
            }
        });

        final LinearLayout quick = new LinearLayout(this);
        quick.setOrientation(LinearLayout.HORIZONTAL);
        final String[][] chips = {
                {"算 17*23", "算一下 17*23 等于多少？"},
                {"现在几点", "现在几点？"},
                {"记住我叫杨雷", "记住：我叫杨雷"},
                // The description prompt is the image path this export was
                // validated on (see helpers/STATUS.md); open-ended "improve it" requests
                // are beyond the Q4 model even though the vision tokens are fine.
                {"描述图片", "请详细描述这张图片的内容。"},
                {"我是谁", "我叫什么名字？我的职业是什么？"},
        };
        for (final String[] chip : chips) {
            final Button button = new Button(this);
            button.setText(chip[0]);
            button.setTextSize(11);
            button.setAllCaps(false);
            button.setOnClickListener(view -> {
                inputView.setText(chip[1]);
                sendTurn();
            });
            quick.addView(button, new LinearLayout.LayoutParams(
                    0, ViewGroup.LayoutParams.WRAP_CONTENT, 1f));
            quickButtons.add(button);
        }
        panel.addView(quick);

        final LinearLayout attachRow = new LinearLayout(this);
        attachRow.setOrientation(LinearLayout.HORIZONTAL);
        attachRow.setGravity(Gravity.CENTER_VERTICAL);
        attachButton = new Button(this);
        attachButton.setText("选择图片");
        attachButton.setTextSize(12);
        attachButton.setAllCaps(false);
        attachButton.setOnClickListener(view -> chooseImage());
        attachRow.addView(attachButton);
        removeButton = new Button(this);
        removeButton.setText("移除图片");
        removeButton.setTextSize(12);
        removeButton.setAllCaps(false);
        removeButton.setEnabled(false);
        removeButton.setOnClickListener(view -> detachImage());
        attachRow.addView(removeButton);
        previewView = new ImageView(this);
        previewView.setAdjustViewBounds(true);
        previewView.setScaleType(ImageView.ScaleType.CENTER_INSIDE);
        attachRow.addView(previewView, new LinearLayout.LayoutParams(dp(64), dp(64)));
        attachLabel = text("未附加图片", 11, COLOR_MUTED, false);
        attachRow.addView(attachLabel);
        panel.addView(attachRow);

        // Runtime switches the user asked for: memory and conversation history.
        // Turning history off gives a clean, fast, context-free chat.
        final LinearLayout switches = new LinearLayout(this);
        switches.setOrientation(LinearLayout.HORIZONTAL);
        switches.setGravity(Gravity.CENTER_VERTICAL);
        final android.widget.Switch memorySwitch = new android.widget.Switch(this);
        memorySwitch.setText("记忆");
        memorySwitch.setTextSize(12);
        memorySwitch.setChecked(true);
        memorySwitch.setOnCheckedChangeListener((view, checked) -> sendOptions());
        final android.widget.Switch historySwitch = new android.widget.Switch(this);
        historySwitch.setText("会话历史");
        historySwitch.setTextSize(12);
        historySwitch.setChecked(true);
        historySwitch.setOnCheckedChangeListener((view, checked) -> sendOptions());
        switches.addView(memorySwitch, new LinearLayout.LayoutParams(
                0, ViewGroup.LayoutParams.WRAP_CONTENT, 1f));
        switches.addView(historySwitch, new LinearLayout.LayoutParams(
                0, ViewGroup.LayoutParams.WRAP_CONTENT, 1f));
        panel.addView(switches);
        memoryToggle = memorySwitch;
        historyToggle = historySwitch;

        inputView = new EditText(this);
        inputView.setHint("问点什么，或让我用工具做事…");
        inputView.setTextSize(14);
        inputView.setMinLines(1);
        inputView.setMaxLines(4);
        inputView.setInputType(InputType.TYPE_CLASS_TEXT
                | InputType.TYPE_TEXT_FLAG_MULTI_LINE
                | InputType.TYPE_TEXT_FLAG_CAP_SENTENCES);
        panel.addView(inputView);

        final LinearLayout actions = new LinearLayout(this);
        actions.setOrientation(LinearLayout.HORIZONTAL);
        sendButton = new Button(this);
        sendButton.setText("发送");
        sendButton.setEnabled(false);
        sendButton.setOnClickListener(view -> {
            if (inferenceRunning) {
                stopTurn();
            } else {
                sendTurn();
            }
        });
        actions.addView(sendButton, new LinearLayout.LayoutParams(
                0, ViewGroup.LayoutParams.WRAP_CONTENT, 1f));
        final Button clearButton = new Button(this);
        clearButton.setText("清屏");
        clearButton.setOnClickListener(view -> {
            // Clearing the screen also drops the pending attachment: a message
            // sent afterwards must not silently carry the earlier image.
            transcript.clear();
            detachImage();
            status("对话已清屏，附加图片已移除");
        });
        actions.addView(clearButton, new LinearLayout.LayoutParams(
                0, ViewGroup.LayoutParams.WRAP_CONTENT, 1f));
        panel.addView(actions);
        return panel;
    }

    private LinearLayout buildModelPanel() {
        final LinearLayout panel = new LinearLayout(this);
        panel.setOrientation(LinearLayout.VERTICAL);
        panel.setVisibility(View.GONE);

        final LinearLayout header = new LinearLayout(this);
        header.setOrientation(LinearLayout.HORIZONTAL);
        final Button refresh = new Button(this);
        refresh.setText("刷新");
        refresh.setAllCaps(false);
        refresh.setOnClickListener(view -> worker.execute(this::refreshEverything));
        header.addView(refresh);
        final Button memoryRefresh = new Button(this);
        memoryRefresh.setText("记忆 / 会话");
        memoryRefresh.setAllCaps(false);
        memoryRefresh.setOnClickListener(view -> worker.execute(() -> {
            refreshMemories();
            refreshSessions();
        }));
        header.addView(memoryRefresh);
        panel.addView(header);

        perfView = text("屏幕开 / 关 基准测试", 12, COLOR_TEXT, true);
        perfView.setPadding(0, dp(6), 0, 0);
        panel.addView(perfView);
        final LinearLayout bench = new LinearLayout(this);
        bench.setOrientation(LinearLayout.HORIZONTAL);
        final Button screenOn = new Button(this);
        screenOn.setText("屏幕常亮基准");
        screenOn.setAllCaps(false);
        screenOn.setOnClickListener(view -> runPerfScenario(true));
        bench.addView(screenOn, new LinearLayout.LayoutParams(
                0, ViewGroup.LayoutParams.WRAP_CONTENT, 1f));
        final Button screenOff = new Button(this);
        screenOff.setText("熄屏基准");
        screenOff.setAllCaps(false);
        screenOff.setOnClickListener(view -> runPerfScenario(false));
        bench.addView(screenOff, new LinearLayout.LayoutParams(
                0, ViewGroup.LayoutParams.WRAP_CONTENT, 1f));
        panel.addView(bench);

        final ScrollView scroll = new ScrollView(this);
        final LinearLayout inside = new LinearLayout(this);
        inside.setOrientation(LinearLayout.VERTICAL);
        runtimeView = text("", 12, COLOR_TEXT, true);
        runtimeView.setTextIsSelectable(true);
        inside.addView(runtimeView);
        toolsList = new LinearLayout(this);
        toolsList.setOrientation(LinearLayout.VERTICAL);
        inside.addView(toolsList);
        logView = text("", 10, COLOR_MUTED, true);
        logView.setTextIsSelectable(true);
        logView.setPadding(0, dp(8), 0, dp(8));
        inside.addView(logView);
        scroll.addView(inside);
        panel.addView(scroll, new LinearLayout.LayoutParams(
                ViewGroup.LayoutParams.MATCH_PARENT, 0, 1f));
        return panel;
    }

    private void showTab(int index) {
        final LinearLayout[] panels = {chatPanel, memoryPanel, sessionPanel, modelPanel};
        for (int position = 0; position < panels.length; position++) {
            panels[position].setVisibility(position == index ? View.VISIBLE : View.GONE);
            tabButtons.get(position).setEnabled(position != index);
        }
        if (index == 1) {
            worker.execute(this::refreshMemories);
        } else if (index == 2) {
            worker.execute(this::refreshSessions);
        }
    }

    // --- runtime startup ----------------------------------------------------

    private void prepareRuntime() {
        // The UI-command watcher is started here (not at the end of the method):
        // prepareRuntime has early returns for a missing runtime or a failed
        // service start, and the hook has to work regardless.
        startUiCommandWatcher();
        final RuntimeInstaller installer =
                new RuntimeInstaller(getAssets(), ROOT, message -> main.post(() -> status(main,
                        statusView, message)));
        final List<String> problems = installer.ensureInstalled();
        if (!problems.isEmpty()) {
            filesReady = false;
            main.post(() -> {
                statusView.setText("运行环境不完整");
                transcript.addError(String.join("\n", problems));
            });
            return;
        }
        main.post(() -> statusView.setText("正在加载 NPU 与 MNN 模型…"));
        try {
            writePlatformConfig();
            // A running service is reused unless this app run replaced the
            // runtime, in which case it has to be restarted to pick it up.
            if (installer.didInstall()) {
                new File(ROOT, "agent/service-ready").delete();
            }
            SERVICE_START.delete();
            try (FileOutputStream ignored = new FileOutputStream(SERVICE_START)) {
            }
            final long deadline = System.currentTimeMillis() + 300_000L;
            final File ready = new File(ROOT, "agent/service-ready");
            while (!ready.isFile()) {
                if (System.currentTimeMillis() > deadline) {
                    throw new IllegalStateException(
                            "运行时服务未启动，请检查 adb root 与 SELinux 标签");
                }
                Thread.sleep(100L);
            }
            filesReady = true;
        } catch (Exception error) {
            Log.e(TAG, "runtime start failed", error);
            filesReady = false;
            main.post(() -> {
                statusView.setText("运行环境启动失败");
                transcript.addError(String.valueOf(error.getMessage()));
            });
            return;
        }

        client.startPolling(new AgentClient.EventListener() {
            @Override
            public void onEvent(JSONObject event) {
                main.post(() -> handleAgentEvent(event));
            }

            @Override
            public void onLogReset() {
                main.post(() -> {
                    transcript.addNotice("运行时已重启，事件流已重置");
                    worker.execute(MainActivity.this::refreshEverything);
                });
            }
        });

        main.post(() -> {
            sendButton.setEnabled(true);
            attachButton.setEnabled(true);
            statusView.setText("模型已就绪 · " + BuildConfig.IMAGE_MODE_LABEL);
        });
        worker.execute(this::refreshEverything);
        main.post(this::sendOptions);
    }

    private void startUiCommandWatcher() {
        final Thread watcher = new Thread(() -> {
            while (!Thread.currentThread().isInterrupted()) {
                try {
                    if (UI_COMMAND.isFile()) {
                        String contents = "";
                        try (RandomAccessFile reader = new RandomAccessFile(UI_COMMAND, "r")) {
                            final byte[] buffer = new byte[(int) reader.length()];
                            reader.readFully(buffer);
                            contents = new String(buffer, StandardCharsets.UTF_8);
                        }
                        UI_COMMAND.delete();
                        final JSONObject command = new JSONObject(contents);
                        android.util.Log.i(TAG, "ui command: " + contents);
                        applyUiCommand(command);
                    }
                    Thread.sleep(250L);
                } catch (InterruptedException interrupted) {
                    Thread.currentThread().interrupt();
                    return;
                } catch (Exception error) {
                    android.util.Log.w(TAG, "ui command failed", error);
                }
            }
        }, "ui-command");
        watcher.setDaemon(true);
        watcher.start();
    }

    /** Finds a visible button by its label and clicks it. */
    private boolean clickButtonByText(String label) {
        final android.view.View root = findViewById(android.R.id.content);
        final java.util.List<android.view.View> queue = new java.util.ArrayList<>();
        queue.add(root);
        while (!queue.isEmpty()) {
            final android.view.View view = queue.remove(0);
            if (view instanceof Button && label.contentEquals(((Button) view).getText())) {
                view.performClick();
                return true;
            }
            if (view instanceof android.view.ViewGroup) {
                final android.view.ViewGroup group = (android.view.ViewGroup) view;
                for (int index = 0; index < group.getChildCount(); index++) {
                    queue.add(group.getChildAt(index));
                }
            }
        }
        return false;
    }

    /** Applies one test command on the main thread, exactly like a user would. */
    private void applyUiCommand(JSONObject command) {
        final String text = command.optString("text", "");
        final boolean send = command.optBoolean("send", true);
        final boolean clear = command.optBoolean("clear", false);
        final String image = command.optString("image", "");
        final boolean pickImage = command.optBoolean("pick_image", false);
        final boolean newSession = command.optBoolean("new_session", false);
        final String chip = command.optString("chip", "");
        final String clickText = command.optString("click_text", "");
        main.post(() -> {
            if (!clickText.isEmpty()) {
                // Click any button in the activity by its label: the UI suite uses
                // this for the buttons that are not quick chips (会话 tab actions,
                // tab switches, the memory view's buttons).
                if (clickButtonByText(clickText)) {
                    return;
                }
                status("没有名为 " + clickText + " 的按钮");
                return;
            }
            if (!chip.isEmpty()) {
                // Click a quick-action chip exactly like a finger would, so the UI
                // suite can test what a button actually sends.
                for (Button button : quickButtons) {
                    if (button.getText().toString().equals(chip)) {
                        button.performClick();
                        return;
                    }
                }
                status("没有名为 " + chip + " 的按钮");
                return;
            }
            if (newSession) {
                startNewSession(command.optString("title", null));
                return;
            }
            if (clear) {
                transcript.clear();
                detachImage();
            }
            if (!image.isEmpty()) {
                final File canvas = new File(image.startsWith("/") ? image
                        : ROOT.getPath() + "/" + image);
                if (canvas.isFile()) {
                    // Same state the picker creates, without a bitmap preview.
                    attachedCanvas = canvas;
                    attachedNote = "ui-command canvas";
                    attachLabel.setText(canvas.getName() + " (ui-command)");
                    if (removeButton != null) {
                        removeButton.setEnabled(true);
                    }
                }
            }
            if (pickImage) {
                chooseImage();
                return;
            }
            if (!text.isEmpty()) {
                inputView.setText(text);
            }
            if (send) {
                if (!filesReady || inferenceRunning) {
                    // The runtime is still loading (or a turn is running): put the
                    // command back instead of dropping it, so an automated run
                    // cannot race the service start.
                    try {
                        writeFileAtomically(UI_COMMAND, command.toString());
                    } catch (Exception ignored) {
                    }
                    android.util.Log.i(TAG, "ui command deferred: filesReady=" + filesReady
                            + " running=" + inferenceRunning);
                } else {
                    sendTurn();
                }
            }
        });
    }

    private void writePlatformConfig() throws IOException {
        try (FileOutputStream output = new FileOutputStream(PLATFORM_CONFIG)) {
            output.write(BuildConfig.VISION_PLATFORM.getBytes(StandardCharsets.US_ASCII));
            output.getFD().sync();
        }
        PLATFORM_CONFIG.setReadable(true, false);
        PLATFORM_CONFIG.setWritable(true, false);
    }

    private void refreshEverything() {
        try {
            final JSONObject capabilities = client.call(new JSONObject("{\"type\":\"capabilities\"}"));
            main.post(() -> renderCapabilities(capabilities));
        } catch (Exception error) {
            Log.w(TAG, "capabilities failed", error);
        }
        try {
            final JSONObject history = client.call(new JSONObject("{\"type\":\"history\"}"));
            main.post(() -> transcript.renderHistory(history.optJSONArray("turns")));
        } catch (Exception error) {
            Log.w(TAG, "history failed", error);
        }
        refreshMemories();
        refreshSessions();
    }

    private void refreshMemories() {
        try {
            final JSONObject response =
                    client.call(new JSONObject("{\"type\":\"list_memories\"}"));
            final JSONArray memories = response.optJSONArray("memories");
            main.post(() -> renderMemories(memories, response.optInt("count", 0)));
        } catch (Exception error) {
            Log.w(TAG, "memory list failed", error);
        }
    }

    private void refreshSessions() {
        try {
            final JSONObject response =
                    client.call(new JSONObject("{\"type\":\"list_sessions\"}"));
            main.post(() -> renderSessions(response.optJSONArray("sessions"),
                    response.optString("current", "")));
        } catch (Exception error) {
            Log.w(TAG, "session list failed", error);
        }
    }

    // --- agent events -------------------------------------------------------

    private void handleAgentEvent(JSONObject event) {
        transcript.handleEvent(event);
        final String type = event.optString("type", "");
        switch (type) {
            case "hello":
                renderCapabilities(event);
                break;
            case "turn_start":
                inferenceStartElapsed = SystemClock.elapsedRealtime();
                logLine("TURN START " + event.optString("text", ""));
                break;
            case "step_end":
                logLine(event.toString());
                break;
            case "memory_write":
                resetLine(logView, "MEMORY WRITE " + event.toString());
                refreshMemories();
                break;
            case "turn_end":
                logLine("TURN END " + event.toString());
                refreshSessions();
                break;
            default:
                break;
        }
        if ("turn_end".equals(type) || "error".equals(type)) {
            appendLog(event.toString());
        }
    }

    private void renderCapabilities(JSONObject capabilities) {
        final JSONObject device = capabilities.optJSONObject("device");
        final JSONObject config = capabilities.optJSONObject("config");
        final JSONObject llm = capabilities.optJSONObject("llm");
        final StringBuilder text = new StringBuilder();
        text.append("模型: ").append(BuildConfig.APP_TITLE).append('\n');
        text.append("平台: ").append(device == null ? "?" : device.optString("platform", "?"))
                .append(" · vision ")
                .append(device == null ? "?" : device.optString("vision", "?")).append('\n');
        text.append("画布: ").append(device == null ? "?" : device.optString("canvas_width"))
                .append('x').append(device == null ? "?" : device.optString("canvas_height"))
                .append(" · visual tokens ")
                .append(device == null ? "?" : device.optString("soft_tokens")).append('\n');
        text.append("线程: ").append(device == null ? "?" : device.optString("threads"))
                .append(" · LLM: ")
                .append(llm == null ? "?" : llm.optString("backend", "?")).append('\n');
        if (config != null) {
            text.append("预算: prompt ").append(config.optInt("context_budget_tokens"))
                    .append(" tok · 历史 ").append(config.optInt("history_budget_tokens"))
                    .append(" tok · 最多工具 ").append(config.optInt("max_tool_calls"))
                    .append(" 次/轮\n");
        }
        text.append("记忆: ").append(capabilities.optInt("memory_records", 0))
                .append(" 条 · 会话: ").append(capabilities.optInt("sessions", 0))
                .append(" · 当前会话: ").append(capabilities.optString("session", "-"));
        runtimeView.setText(text.toString());

        toolsList.removeAllViews();
        final JSONArray functions = capabilities.optJSONArray("functions");
        toolsList.addView(text("原生工具 (" + (functions == null ? 0 : functions.length())
                + ")", 13, COLOR_TEXT, true));
        for (int index = 0; functions != null && index < functions.length(); index++) {
            final JSONObject function = functions.optJSONObject(index);
            if (function == null) {
                continue;
            }
            final TextView view = text("• " + function.optString("name", "")
                    + " — " + function.optString("description", ""), 11, COLOR_MUTED, false);
            view.setPadding(0, dp(2), 0, dp(2));
            toolsList.addView(view);
        }
        sendButton.setEnabled(filesReady && !inferenceRunning);
    }

    // --- chat ---------------------------------------------------------------

    private void sendTurn() {
        if (!filesReady || inferenceRunning) {
            return;
        }
        String text = inputView.getText().toString().trim();
        if (text.isEmpty() && attachedCanvas == null) {
            return;
        }
        if (text.isEmpty()) {
            text = BuildConfig.FIXED_PROMPT;
        }
        try {
            final JSONObject request = new JSONObject();
            request.put("type", "turn");
            request.put("text", text);
            if (attachedCanvas != null) {
                request.put("image", attachedCanvas.getName());
            }
            transcript.beginTurn(text, attachedCanvas != null, attachedNote);
            client.send(request);
            // The image now lives in the conversation; the next message must not
            // carry it again (that used to surprise: a greeting answered about an
            // image).  Follow-up questions still resolve through the session
            // history, which keeps the image turn and its cached canvas.
            detachImage();
            inputView.setText("");
            inferenceRunning = true;
            sendButton.setText("停止");
            acquireWakeLock();
            getWindow().addFlags(WindowManager.LayoutParams.FLAG_KEEP_SCREEN_ON);
        } catch (Exception error) {
            transcript.addError("发送失败: " + error.getMessage());
        }
    }

    private void stopTurn() {
        client.cancelCurrentTurn();
        status("已请求停止当前回合");
    }

    private boolean turnRunning() {
        return inferenceRunning;
    }

    private void finishTurn(String message) {
        inferenceRunning = false;
        sendButton.setText("发送");
        sendButton.setEnabled(filesReady);
        releaseWakeLocks();
        getWindow().clearFlags(WindowManager.LayoutParams.FLAG_KEEP_SCREEN_ON);
        status(message);
    }

    private void writeFileAtomically(File target, String contents) throws IOException {
        final File temporary = new File(target.getPath() + ".part");
        try (FileOutputStream output = new FileOutputStream(temporary)) {
            output.write(contents.getBytes(StandardCharsets.UTF_8));
            output.getFD().sync();
        }
        if (target.exists() && !target.delete()) {
            throw new IOException("cannot replace " + target);
        }
        if (!temporary.renameTo(target)) {
            throw new IOException("cannot publish " + target);
        }
        target.setReadable(true, false);
        target.setWritable(true, false);
    }

    /** Pushes the memory / history switches to the native runtime. */
    private void sendOptions() {
        final boolean memory = memoryToggle == null || memoryToggle.isChecked();
        final boolean history = historyToggle == null || historyToggle.isChecked();
        worker.execute(() -> {
            try {
                final JSONObject request = new JSONObject();
                request.put("type", "config");
                request.put("memory", memory);
                request.put("history", history);
                client.call(request);
                main.post(() -> status("记忆：" + (memory ? "开" : "关")
                        + " · 会话历史：" + (history ? "开" : "关")));
            } catch (Exception error) {
                Log.w(TAG, "config failed", error);
            }
        });
    }

    /** Drops the pending attachment (清屏, 移除图片, or after a turn was sent). */
    private void detachImage() {
        attachedCanvas = null;
        attachedNote = null;
        if (previewView != null) {
            previewView.setImageDrawable(null);
        }
        if (attachLabel != null) {
            attachLabel.setText("未附加图片");
        }
        if (removeButton != null) {
            removeButton.setEnabled(false);
        }
    }

    private void chooseImage() {
        final Intent intent = new Intent(Intent.ACTION_OPEN_DOCUMENT);
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
        final Uri uri = data.getData();
        status("正在准备图像张量…");
        worker.execute(() -> {
            try {
                final ImageDecoder.Source source =
                        ImageDecoder.createSource(getContentResolver(), uri);
                final Bitmap bitmap = ImageDecoder.decodeBitmap(source,
                        (decoder, info, src) -> decoder.setAllocator(ImageDecoder.ALLOCATOR_SOFTWARE));
                final CanvasLetterbox.Result result = letterbox.prepare(bitmap, BRIDGE_INPUT);
                attachedCanvas = BRIDGE_INPUT;
                attachedNote = String.format(Locale.US, "%dx%d 画布 · %d visual tokens",
                        result.preview.getWidth(), result.preview.getHeight(), result.softTokens);
                main.post(() -> {
                    previewView.setImageBitmap(result.preview);
                    attachLabel.setText(attachedNote);
                    if (removeButton != null) {
                        removeButton.setEnabled(true);
                    }
                    status("图片已就绪，" + result.softTokens + " 个视觉 token（发送后自动移除附加）");
                });
            } catch (Exception error) {
                main.post(() -> {
                    status("图像准备失败: " + error.getMessage());
                    transcript.addError("图像准备失败: " + error.getMessage());
                });
            }
        });
    }

    // --- memory / sessions --------------------------------------------------

    private void renderMemories(JSONArray memories, int count) {
        memoryList.removeAllViews();
        final LinearLayout actions = new LinearLayout(this);
        actions.setOrientation(LinearLayout.HORIZONTAL);
        final Button clear = new Button(this);
        clear.setText("清空非置顶 (" + count + ")");
        clear.setAllCaps(false);
        clear.setOnClickListener(view -> worker.execute(() -> {
            try {
                final JSONObject request = new JSONObject();
                request.put("type", "clear_memories");
                request.put("keep_pinned", true);
                client.call(request);
            } catch (Exception error) {
                Log.w(TAG, "clear memories failed", error);
            }
            refreshMemories();
        }));
        actions.addView(clear, new LinearLayout.LayoutParams(
                0, ViewGroup.LayoutParams.WRAP_CONTENT, 1f));
        // One tap to forget everything the agent stored about the user.
        final Button clearAll = new Button(this);
        clearAll.setText("清空全部记忆");
        clearAll.setAllCaps(false);
        clearAll.setOnClickListener(view -> new AlertDialog.Builder(this)
                .setTitle("清空全部记忆？")
                .setMessage("会删除长期记忆库里的所有条目（含置顶）。会话历史不受影响。")
                .setPositiveButton("清空", (dialog, which) -> worker.execute(() -> {
                    try {
                        final JSONObject request = new JSONObject();
                        request.put("type", "clear_memories");
                        request.put("keep_pinned", false);
                        client.call(request);
                    } catch (Exception error) {
                        Log.w(TAG, "clear all memories failed", error);
                    }
                    refreshMemories();
                    main.post(() -> status("记忆库已清空"));
                }))
                .setNegativeButton("取消", null)
                .show());
        actions.addView(clearAll, new LinearLayout.LayoutParams(
                0, ViewGroup.LayoutParams.WRAP_CONTENT, 1f));
        memoryList.addView(actions);
        if (memories == null || memories.length() == 0) {
            memoryList.addView(text("记忆库为空。对助手说“记住…”即可写入。", 12, COLOR_MUTED, false));
            return;
        }
        for (int index = 0; index < memories.length(); index++) {
            final JSONObject memory = memories.optJSONObject(index);
            if (memory == null) {
                continue;
            }
            final LinearLayout row = new LinearLayout(this);
            row.setOrientation(LinearLayout.VERTICAL);
            row.setBackground(rounded(COLOR_PANEL, 8));
            row.setPadding(dp(10), dp(8), dp(10), dp(8));
            final LinearLayout.LayoutParams params = new LinearLayout.LayoutParams(
                    ViewGroup.LayoutParams.MATCH_PARENT, ViewGroup.LayoutParams.WRAP_CONTENT);
            params.topMargin = dp(6);
            row.setLayoutParams(params);
            row.addView(text("[" + memory.optString("kind", "note") + " "
                            + memory.optString("id", "") + "] pinned="
                            + memory.optBoolean("pinned", false) + " importance="
                            + String.format(Locale.US, "%.2f", memory.optDouble("importance", 0.5)),
                    11, COLOR_MUTED, true));
            row.addView(text(memory.optString("text", ""), 13, COLOR_TEXT, false));
            final Button delete = new Button(this);
            delete.setText("删除");
            delete.setAllCaps(false);
            delete.setOnClickListener(view -> worker.execute(() -> {
                try {
                    final JSONObject request = new JSONObject();
                    request.put("type", "delete_memory");
                    request.put("id", memory.optString("id", ""));
                    client.call(request);
                } catch (Exception error) {
                    Log.w(TAG, "delete memory failed", error);
                }
                refreshMemories();
            }));
            row.addView(delete);
            memoryList.addView(row);
        }
    }

    private void renderSessions(JSONArray sessions, String current) {
        sessionList.removeAllViews();
        final LinearLayout sessionActions = new LinearLayout(this);
        sessionActions.setOrientation(LinearLayout.HORIZONTAL);
        final Button create = new Button(this);
        create.setText("新建会话");
        create.setAllCaps(false);
        create.setOnClickListener(view -> startNewSession(null));
        sessionActions.addView(create, new LinearLayout.LayoutParams(
                0, ViewGroup.LayoutParams.WRAP_CONTENT, 1f));
        // One tap to wipe the whole conversation history (memory is separate).
        final Button deleteAll = new Button(this);
        deleteAll.setText("删除全部会话");
        deleteAll.setAllCaps(false);
        deleteAll.setOnClickListener(view -> new AlertDialog.Builder(this)
                .setTitle("删除全部会话？")
                .setMessage("会删除所有会话及其历史记录，并新建一个空会话。长期记忆不受影响。")
                .setPositiveButton("删除", (dialog, which) -> worker.execute(() -> {
                    try {
                        final JSONObject request = new JSONObject();
                        request.put("type", "delete_all_sessions");
                        final JSONObject response = client.call(request);
                        main.post(() -> {
                            transcript.clear();
                            status("已删除全部会话，当前会话 "
                                    + response.optString("session", ""));
                        });
                    } catch (Exception error) {
                        Log.w(TAG, "delete all sessions failed", error);
                    }
                    refreshSessions();
                }))
                .setNegativeButton("取消", null)
                .show());
        sessionActions.addView(deleteAll, new LinearLayout.LayoutParams(
                0, ViewGroup.LayoutParams.WRAP_CONTENT, 1f));
        sessionList.addView(sessionActions);
        if (sessions == null || sessions.length() == 0) {
            sessionList.addView(text("还没有会话。", 12, COLOR_MUTED, false));
            return;
        }
        for (int index = 0; index < sessions.length(); index++) {
            final JSONObject session = sessions.optJSONObject(index);
            if (session == null) {
                continue;
            }
            final String id = session.optString("id", "");
            final LinearLayout row = new LinearLayout(this);
            row.setOrientation(LinearLayout.VERTICAL);
            row.setBackground(rounded(COLOR_PANEL, 8));
            row.setPadding(dp(10), dp(8), dp(10), dp(8));
            final LinearLayout.LayoutParams params = new LinearLayout.LayoutParams(
                    ViewGroup.LayoutParams.MATCH_PARENT, ViewGroup.LayoutParams.WRAP_CONTENT);
            params.topMargin = dp(6);
            row.setLayoutParams(params);
            row.addView(text((id.equals(current) ? "▶ " : "  ") + id + " · "
                            + session.optInt("turns", 0) + " 轮 · 压缩 "
                            + session.optInt("compactions", 0) + " 次",
                    11, id.equals(current) ? COLOR_ACCENT : COLOR_MUTED, true));
            row.addView(text(session.optString("title", ""), 13, COLOR_TEXT, false));
            final LinearLayout actions = new LinearLayout(this);
            actions.setOrientation(LinearLayout.HORIZONTAL);
            final Button open = new Button(this);
            open.setText("打开");
            open.setAllCaps(false);
            open.setOnClickListener(view -> worker.execute(() -> openSession(id)));
            actions.addView(open, new LinearLayout.LayoutParams(
                    0, ViewGroup.LayoutParams.WRAP_CONTENT, 1f));
            final Button rename = new Button(this);
            rename.setText("重命名");
            rename.setAllCaps(false);
            rename.setOnClickListener(view -> promptRename(id));
            actions.addView(rename, new LinearLayout.LayoutParams(
                    0, ViewGroup.LayoutParams.WRAP_CONTENT, 1f));
            final Button delete = new Button(this);
            delete.setText("删除");
            delete.setAllCaps(false);
            delete.setOnClickListener(view -> worker.execute(() -> {
                try {
                    final JSONObject request = new JSONObject();
                    request.put("type", "delete_session");
                    request.put("session", id);
                    client.call(request);
                } catch (Exception error) {
                    Log.w(TAG, "delete session failed", error);
                }
                refreshSessions();
            }));
            actions.addView(delete, new LinearLayout.LayoutParams(
                    0, ViewGroup.LayoutParams.WRAP_CONTENT, 1f));
            row.addView(actions);
            sessionList.addView(row);
        }
    }

    private void openSession(String id) {
        try {
            final JSONObject request = new JSONObject();
            request.put("type", "open_session");
            request.put("session", id);
            client.call(request);
            final JSONObject history = client.call(new JSONObject("{\"type\":\"history\"}"));
            main.post(() -> {
                transcript.renderHistory(history.optJSONArray("turns"));
                showTab(0);
                status("已切换到会话 " + id);
            });
            refreshSessions();
        } catch (Exception error) {
            main.post(() -> status("打开会话失败: " + error.getMessage()));
        }
    }

    private void startNewSession(String title) {
        worker.execute(() -> {
            try {
                final JSONObject request = new JSONObject();
                request.put("type", "new_session");
                if (title != null) {
                    request.put("title", title);
                }
                final JSONObject response = client.call(request);
                main.post(() -> {
                    transcript.clear();
                    showTab(0);
                    status("新会话 " + response.optString("session", ""));
                });
                refreshSessions();
            } catch (Exception error) {
                main.post(() -> status("新建会话失败: " + error.getMessage()));
            }
        });
    }

    private void promptRename(String id) {
        final EditText field = new EditText(this);
        field.setHint("会话标题");
        new AlertDialog.Builder(this)
                .setTitle("重命名会话 " + id)
                .setView(field)
                .setPositiveButton("保存", (dialog, which) -> worker.execute(() -> {
                    try {
                        final JSONObject request = new JSONObject();
                        request.put("type", "rename_session");
                        request.put("session", id);
                        request.put("title", field.getText().toString());
                        client.call(request);
                    } catch (Exception error) {
                        Log.w(TAG, "rename failed", error);
                    }
                    refreshSessions();
                }))
                .setNegativeButton("取消", null)
                .show();
    }

    // --- benchmark (legacy single-shot path) --------------------------------

    private void runPerfScenario(boolean screenOn) {
        if (!filesReady || attachedCanvas == null || inferenceRunning) {
            status("请先准备图片并结束当前回合");
            return;
        }
        final String label = screenOn ? "SCREEN_ON" : "SCREEN_OFF";
        perfMonitor = new PerfMonitor(screenOn, label);
        decodeTokenTotal = 0;
        lastTokenIndex = -1;
        firstDecodeSampleElapsed = 0L;
        openPerfLog(label);
        applyScreenStateForBenchmark(screenOn);
        worker.execute(() -> {
            try {
                if (screenOn) {
                    perfMonitor.begin();
                } else {
                    final long deadline = System.currentTimeMillis() + 60_000L;
                    while (System.currentTimeMillis() < deadline
                            && !Thread.currentThread().isInterrupted()) {
                        Thread.sleep(500L);
                        perfMonitor.sample("warmup-idle");
                    }
                    perfMonitor.begin();
                }
                runLegacyRequest();
            } catch (Exception error) {
                Log.e(TAG, "benchmark failed", error);
                main.post(() -> perfView.setText("基准失败: " + error.getMessage()));
            } finally {
                perfMonitor.finish();
                closePerfLog();
                releaseWakeLocks();
                main.post(() -> getWindow()
                        .clearFlags(WindowManager.LayoutParams.FLAG_KEEP_SCREEN_ON));
            }
        });
    }

    private void runLegacyRequest() throws Exception {
        final String prompt = BuildConfig.FIXED_PROMPT;
        try (FileWriter writer = new FileWriter(BRIDGE_PROMPT)) {
            writer.write(prompt);
        }
        BRIDGE_STOP.delete();
        inferenceRunning = true;
        main.post(() -> {
            sendButton.setText("停止");
            status("基准测试运行中 (" + prompt + ")");
        });
        final long requestedAt = System.currentTimeMillis();
        try (FileOutputStream ignored = new FileOutputStream(BRIDGE_REQUEST)) {
        }
        inferenceStartElapsed = SystemClock.elapsedRealtime();
        final long deadline = System.currentTimeMillis() + 300_000L;
        while (!BRIDGE_LOG.isFile() || BRIDGE_LOG.lastModified() < requestedAt) {
            if (System.currentTimeMillis() > deadline) {
                throw new IllegalStateException("bridge not running");
            }
            Thread.sleep(50L);
        }
        final StringBuilder report = new StringBuilder();
        try (RandomAccessFile log = new RandomAccessFile(BRIDGE_LOG, "r")) {
            while (true) {
                final String line = log.readLine();
                if (line == null) {
                    Thread.sleep(50L);
                    continue;
                }
                final String text = new String(line.getBytes(StandardCharsets.ISO_8859_1),
                        StandardCharsets.UTF_8);
                if (text.startsWith("TOKEN ")) {
                    handleBenchmarkToken(text);
                    continue;
                }
                if (text.startsWith("FIRST_TOKEN") || text.startsWith("RESULT ")) {
                    report.append(text).append('\n');
                    main.post(() -> perfView.setText(report.toString()));
                }
                if (text.startsWith("RESULT request_complete=") || text.startsWith("ERROR ")) {
                    break;
                }
            }
        }
        inferenceRunning = false;
        main.post(() -> finishTurn("基准测试完成"));
    }

    private void handleBenchmarkToken(String line) {
        final int index = parseInt(field(line, "index"));
        final long now = SystemClock.elapsedRealtime();
        if (lastTokenIndex >= 0 && index > lastTokenIndex) {
            final long interval = now - lastTokenElapsed;
            if (interval > 0) {
                decodeTokenTotal += index - lastTokenIndex;
                if (firstDecodeSampleElapsed == 0L) {
                    firstDecodeSampleElapsed = now;
                }
                final double tps = (index - lastTokenIndex) * 1000.0 / interval;
                final String text = String.format(Locale.US, "token interval %.0f ms → %.2f tok/s",
                        (double) interval, tps);
                main.post(() -> perfView.setText(text));
                logLine(text);
            }
        }
        lastTokenIndex = index;
        lastTokenElapsed = now;
    }

    private void applyScreenStateForBenchmark(boolean screenOn) {
        final PowerManager manager = (PowerManager) getSystemService(POWER_SERVICE);
        if (screenOn) {
            getWindow().addFlags(WindowManager.LayoutParams.FLAG_KEEP_SCREEN_ON);
            if (manager != null && !manager.isInteractive()) {
                benchmarkWakeLock = manager.newWakeLock(
                        PowerManager.SCREEN_BRIGHT_WAKE_LOCK
                                | PowerManager.ACQUIRE_CAUSES_WAKEUP, "Gemma4Agent:bench-on");
                benchmarkWakeLock.acquire();
            }
        } else {
            getWindow().clearFlags(WindowManager.LayoutParams.FLAG_KEEP_SCREEN_ON);
            if (manager != null && manager.isInteractive()) {
                benchmarkWakeLock = manager.newWakeLock(
                        PowerManager.PROXIMITY_SCREEN_OFF_WAKE_LOCK, "Gemma4Agent:bench-off");
                benchmarkWakeLock.acquire(10_000L);
                try {
                    Thread.sleep(500L);
                } catch (InterruptedException ignored) {
                    Thread.currentThread().interrupt();
                }
            }
        }
    }

    // --- helpers ------------------------------------------------------------

    private void acquireWakeLock() {
        final PowerManager manager = (PowerManager) getSystemService(POWER_SERVICE);
        if (manager != null && (turnWakeLock == null || !turnWakeLock.isHeld())) {
            turnWakeLock = manager.newWakeLock(PowerManager.PARTIAL_WAKE_LOCK, "Gemma4Agent:turn");
            turnWakeLock.acquire(600_000L);
        }
    }

    private void releaseWakeLocks() {
        if (turnWakeLock != null && turnWakeLock.isHeld()) {
            turnWakeLock.release();
        }
        if (benchmarkWakeLock != null && benchmarkWakeLock.isHeld()) {
            benchmarkWakeLock.release();
        }
    }

    private void openPerfLog(String label) {
        final File directory = new File(ROOT, "perf");
        if (!directory.isDirectory() && !directory.mkdirs()) {
            return;
        }
        final File target = new File(directory,
                "perf-" + label + "-" + System.currentTimeMillis() + ".log");
        try {
            perfLogWriter = new PrintWriter(new FileWriter(target, true));
        } catch (IOException error) {
            Log.w(TAG, "cannot open the perf log", error);
        }
    }

    private void closePerfLog() {
        if (perfLogWriter != null) {
            perfLogWriter.flush();
            perfLogWriter.close();
            perfLogWriter = null;
        }
    }

    private void logLine(String line) {
        if (perfLogWriter != null) {
            perfLogWriter.println(line);
            perfLogWriter.flush();
        }
        Log.i(TAG, line);
    }

    private void appendLog(String line) {
        if (logView == null) {
            return;
        }
        final String current = logView.getText().toString();
        final String[] lines = current.split("\n");
        final StringBuilder builder = new StringBuilder();
        final int keep = 40;
        for (int index = Math.max(0, lines.length - keep); index < lines.length; index++) {
            builder.append(lines[index]).append('\n');
        }
        builder.append(line).append('\n');
        logView.setText(builder.toString());
    }

    private void resetLine(TextView view, String line) {
        appendLog(line);
    }

    private void status(String message) {
        statusView.setText(message);
    }

    private static void status(Handler handler, TextView view, String message) {
        handler.post(() -> view.setText(message));
    }

    private TextView text(String value, int sizeSp, int color, boolean bold) {
        final TextView view = new TextView(this);
        view.setText(value);
        view.setTextSize(TypedValue.COMPLEX_UNIT_SP, sizeSp);
        view.setTextColor(color);
        if (bold) {
            view.setTypeface(Typeface.DEFAULT, Typeface.BOLD);
        }
        return view;
    }

    private GradientDrawable rounded(int color, int radiusDp) {
        final GradientDrawable drawable = new GradientDrawable();
        drawable.setColor(color);
        drawable.setCornerRadius(dp(radiusDp));
        return drawable;
    }

    private int dp(int value) {
        return Math.round(value * getResources().getDisplayMetrics().density);
    }

    private static int parseInt(String value) {
        try {
            return Integer.parseInt(value);
        } catch (NumberFormatException ignored) {
            return -1;
        }
    }

    private static String field(String line, String name) {
        final String prefix = name + "=";
        for (final String part : line.split(" ")) {
            if (part.startsWith(prefix)) {
                return part.substring(prefix.length());
            }
        }
        return "?";
    }
}
