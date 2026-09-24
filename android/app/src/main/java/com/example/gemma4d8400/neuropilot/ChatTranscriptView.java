package com.example.gemma4d8400.neuropilot;

import android.content.Context;
import android.graphics.Color;
import android.graphics.Typeface;
import android.graphics.drawable.GradientDrawable;
import android.text.util.Linkify;
import android.util.Base64;
import android.util.TypedValue;
import android.view.Gravity;
import android.view.View;
import android.view.ViewGroup;
import android.widget.LinearLayout;
import android.widget.ScrollView;
import android.widget.TextView;

import org.json.JSONArray;
import org.json.JSONObject;

import java.util.Locale;

/**
 * Renders the agent's event stream as a chat transcript.
 *
 * <p>Everything the agent does is visible: the memory it recalled, the plan it
 * follows, every tool call with its arguments and result, the context budget it
 * is using, and the per-turn metrics (TTFT, tokens, tokens/second).  The view
 * is a pure renderer - it never decides anything about the agent.
 */
public final class ChatTranscriptView {

    public interface Listener {
        void onTurnFinished(boolean ok);

        /** Progress line for the status bar (e.g. "prefill 900 tok …"). */
        void onStatus(String message);
    }

    private static final int COLOR_USER = 0xff1b6ef3;
    private static final int COLOR_ASSISTANT = 0xfff2f3f5;
    private static final int COLOR_CARD = 0xfffbfbfc;
    private static final int COLOR_CARD_BORDER = 0xffe2e5ea;
    private static final int COLOR_TEXT = 0xff14161a;
    private static final int COLOR_MUTED = 0xff5f6673;
    private static final int COLOR_ACCENT = 0xff0a7d55;
    private static final int COLOR_WARN = 0xffb3541e;

    private final Context context;
    private final LinearLayout container;
    private final ScrollView scroll;
    private final StreamingUtf8Decoder decoder = new StreamingUtf8Decoder();
    private final float density;
    private Listener listener;
    private boolean autoScroll = true;

    private LinearLayout currentTurn;
    private LinearLayout currentSteps;
    private LinearLayout planCard;
    private TextView planBody;
    private TextView assistantBubble;
    private TextView metricsLine;
    private TextView liveStep;
    private LinearLayout liveCard;
    private LinearLayout lastToolCard;
    private StringBuilder answerBuffer;
    private boolean turnRunning;

    /** Events that belong to a turn and therefore need a turn block. */
    private static final java.util.Set<String> TURN_SCOPED_EVENTS = new java.util.HashSet<>(
            java.util.Arrays.asList("turn_start", "recall", "image", "context", "plan",
                    "step_start", "step_delta", "step_end", "tool_call", "tool_result",
                    "tool_alias", "memory_write", "context_compacted", "answer", "metrics",
                    "turn_end", "error"));
    private static final java.util.Set<String> turnStartTexts = new java.util.HashSet<>(
            java.util.Arrays.asList("turn_start", "step_delta", "step_end", "answer",
                    "metrics", "turn_end"));

    private static boolean opensTurnBlock(String type) {
        return TURN_SCOPED_EVENTS.contains(type);
    }

    public ChatTranscriptView(Context context, LinearLayout container, ScrollView scroll) {
        this.context = context;
        this.container = container;
        this.scroll = scroll;
        this.density = context.getResources().getDisplayMetrics().density;
        scroll.setOnScrollChangeListener((view, x, y, oldX, oldY) -> {
            final int remaining = scroll.getChildAt(0) == null
                    ? 0
                    : scroll.getChildAt(0).getBottom() - (scroll.getHeight() + scroll.getScrollY());
            autoScroll = remaining < dp(120);
        });
    }

    public void setListener(Listener listener) {
        this.listener = listener;
    }

    public boolean isTurnRunning() {
        return turnRunning;
    }

    public void clear() {
        container.removeAllViews();
        currentTurn = null;
        currentSteps = null;
        planCard = null;
        planBody = null;
        assistantBubble = null;
        metricsLine = null;
        liveCard = null;
        liveStep = null;
        lastToolCard = null;
        answerBuffer = null;
        turnRunning = false;
    }

    public void addNotice(String text) {
        final TextView view = label(text, 13, COLOR_MUTED, false);
        view.setPadding(0, dp(6), 0, dp(6));
        container.addView(view);
        scrollDown();
    }

    public void addError(String text) {
        final TextView view = label("⚠ " + text, 13, COLOR_WARN, false);
        view.setPadding(0, dp(6), 0, dp(6));
        container.addView(view);
        scrollDown();
    }

    /** Starts a new turn block; the events that follow fill it in. */
    public void beginTurn(String userText, boolean hasImage, String imageNote) {
        currentTurn = new LinearLayout(context);
        currentTurn.setOrientation(LinearLayout.VERTICAL);
        currentTurn.setLayoutParams(new LinearLayout.LayoutParams(
                ViewGroup.LayoutParams.MATCH_PARENT, ViewGroup.LayoutParams.WRAP_CONTENT));
        container.addView(currentTurn);

        final LinearLayout userRow = new LinearLayout(context);
        userRow.setOrientation(LinearLayout.HORIZONTAL);
        userRow.setGravity(Gravity.END);
        final TextView bubble = label(userText, 15, Color.WHITE, false);
        bubble.setBackground(rounded(COLOR_USER, 14));
        bubble.setPadding(dp(12), dp(9), dp(12), dp(9));
        final LinearLayout.LayoutParams bubbleParams = new LinearLayout.LayoutParams(
                ViewGroup.LayoutParams.WRAP_CONTENT, ViewGroup.LayoutParams.WRAP_CONTENT);
        bubbleParams.leftMargin = dp(40);
        bubbleParams.topMargin = dp(10);
        userRow.addView(bubble, bubbleParams);
        currentTurn.addView(userRow);

        if (hasImage && imageNote != null && !imageNote.isEmpty()) {
            final TextView chip = label("🖼 " + imageNote, 12, COLOR_MUTED, false);
            chip.setBackground(rounded(COLOR_CARD, 10));
            chip.setPadding(dp(10), dp(6), dp(10), dp(6));
            final LinearLayout.LayoutParams params = new LinearLayout.LayoutParams(
                    ViewGroup.LayoutParams.WRAP_CONTENT, ViewGroup.LayoutParams.WRAP_CONTENT);
            params.topMargin = dp(6);
            currentTurn.addView(chip, params);
        }

        currentSteps = new LinearLayout(context);
        currentSteps.setOrientation(LinearLayout.VERTICAL);
        currentTurn.addView(currentSteps);

        assistantBubble = label("", 15, COLOR_TEXT, false);
        assistantBubble.setTextIsSelectable(true);
        assistantBubble.setBackground(rounded(COLOR_ASSISTANT, 14));
        assistantBubble.setPadding(dp(12), dp(9), dp(12), dp(9));
        final LinearLayout.LayoutParams answerParams = new LinearLayout.LayoutParams(
                ViewGroup.LayoutParams.WRAP_CONTENT, ViewGroup.LayoutParams.WRAP_CONTENT);
        answerParams.rightMargin = dp(32);
        answerParams.topMargin = dp(8);
        currentTurn.addView(assistantBubble, answerParams);

        metricsLine = label("", 11, COLOR_MUTED, true);
        metricsLine.setPadding(dp(2), dp(4), 0, dp(2));
        currentTurn.addView(metricsLine);

        planCard = null;
        planBody = null;
        liveCard = null;
        liveStep = null;
        lastToolCard = null;
        answerBuffer = new StringBuilder();
        turnRunning = true;
        decoder.reset();
        scrollDown();
    }

    /** Applies one agent event.  Must be called on the UI thread. */
    public void handleEvent(JSONObject event) {
        final String type = event.optString("type", "");
        // The app also replays the tail of the event log on start, and turns can
        // be started by another client (adb, a previous app run).  Any turn
        // scoped event without an open turn block has to open one, otherwise the
        // renderer would add views to a null container.
        if (currentTurn == null && opensTurnBlock(type)) {
            final String text = event.has("text") ? event.optString("text")
                    : (turnStartTexts.contains(type) ? "(" + type + ")" : "(脚本发起的轮次)");
            beginTurn(text, event.optBoolean("has_image", false),
                    event.optString("note", ""));
            turnRunning = false;   // it is history until a turn_end says otherwise
            if ("turn_start".equals(type)) {
                scrollDown();
                return;
            }
        }
        switch (type) {
            case "hello":
                addNotice("Agent 已就绪: " + event.optString("session", "-")
                        + " / 工具 " + optLength(event, "functions") + " 个");
                break;
            case "intent":
                // The router's decision, so the user can see *why* this turn
                // consults memory or expects a tool.
                addStep("意图理解 · " + event.optString("label", ""),
                        String.format(Locale.US, "kind=%s confidence=%.2f%s%s%s",
                                event.optString("kind", ""),
                                event.optDouble("confidence", 0.0),
                                event.optBoolean("memory", false) ? " · 检索记忆" : " · 不检索记忆",
                                event.optBoolean("tools", false) ? " · 允许工具" : " · 纯文本回答",
                                event.optString("preferred_tool", "").isEmpty()
                                        ? "" : " · 期望 " + event.optString("preferred_tool")),
                        COLOR_MUTED);
                break;
            case "session_notes":
                // The Notetaker digest: what this session established so far.
                addStep("会话笔记 · " + event.optInt("turn", 0) + " 轮",
                        event.optString("notes", "").trim(), COLOR_MUTED);
                break;
            case "recall": {
                final JSONArray items = event.optJSONArray("items");
                final int count = event.optInt("count", 0);
                if (count <= 0) {
                    addStep("记忆召回", "没有匹配的长期记忆 (库内 "
                            + event.optInt("records", 0) + " 条)", COLOR_MUTED);
                    break;
                }
                final StringBuilder text = new StringBuilder();
                for (int index = 0; items != null && index < items.length(); index++) {
                    final JSONObject item = items.optJSONObject(index);
                    if (item == null) {
                        continue;
                    }
                    text.append("• [").append(item.optString("kind", "note")).append(" ")
                            .append(String.format(Locale.US, "%.2f", item.optDouble("score", 0.0)))
                            .append("] ").append(item.optString("text", "")).append('\n');
                }
                addStep("记忆召回 (" + count + ")", text.toString().trim(), COLOR_ACCENT);
                break;
            }
            case "image":
                addStep("图像输入", event.optInt("canvas_width", 0) + "x"
                        + event.optInt("canvas_height", 0) + " canvas, "
                        + event.optInt("tokens", 0) + " visual tokens\n"
                        + event.optString("sha256", "").substring(0,
                                Math.min(16, event.optString("sha256", "").length())),
                        COLOR_MUTED);
                break;
            case "context":
                addStep("上下文预算", "tokens " + event.optInt("tokens", 0) + "/"
                        + event.optInt("budget", 0) + " · history "
                        + event.optInt("history_tokens", 0) + "/"
                        + event.optInt("history_budget", 0)
                        + (event.optBoolean("compacted", false) ? " · 触发压缩" : ""),
                        COLOR_MUTED);
                break;
            case "context_compacted":
                addStep("历史压缩", "已把第 1-" + event.optInt("upto_turn", 0)
                        + " 轮折叠成摘要:\n" + event.optString("summary", ""), COLOR_WARN);
                break;
            case "plan":
                renderPlan(event);
                break;
            case "step_start":
                // The live card is created lazily by the first step_delta so an
                // empty step does not leave a stray box behind.
                liveCard = null;
                liveStep = null;
                if (listener != null) {
                    // The prefill of a fresh session takes seconds on the phone,
                    // so show what the runner is doing while nothing streams yet.
                    listener.onStatus(String.format(Locale.US,
                            "步骤 %d · %s · 上下文 %d tokens",
                            event.optInt("step", 0),
                            "full".equals(event.optString("mode", ""))
                                    ? "全量 prefill" : "KV 复用增量",
                            event.optInt("context_tokens", 0)));
                }
                break;
            case "step_delta":
                appendDelta(event);
                break;
            case "step_end":
                finishStep(event);
                break;
            case "tool_call":
                openToolCard(event);
                break;
            case "tool_result":
                closeToolCard(event);
                break;
            case "memory_write":
                addStep(memoryActionLabel(event) + " " + event.optString("id", ""),
                        "[" + event.optString("kind", "") + "] " + event.optString("text", "")
                                + memoryActionNote(event),
                        COLOR_ACCENT);
                break;
            case "answer":
                if (assistantBubble != null) {
                    assistantBubble.setText(event.optString("text", ""));
                }
                break;
            case "metrics":
                if (metricsLine != null) {
                    metricsLine.setText(String.format(Locale.US,
                            "TTFT %.0f ms · %.1f tok/s · prompt %d · gen %d · ctx %d/%d%s",
                            event.optDouble("ttft_ms", 0.0),
                            event.optDouble("decode_tokens_per_second", 0.0),
                            event.optLong("prompt_tokens", 0),
                            event.optLong("generated_tokens", 0),
                            event.optInt("context_tokens", 0),
                            event.optInt("context_budget", 0),
                            event.optBoolean("cache_hit", false)
                                    ? " · KV 复用 " + event.optInt("cache_delta_tokens", 0) + " tok"
                                    : " · 全量 prefill"));
                }
                break;
            case "turn_end":
                turnRunning = false;
                if (listener != null) {
                    listener.onTurnFinished(event.optBoolean("ok", true));
                }
                break;
            case "error":
                addError(event.optString("message", "unknown error")
                        + (event.optString("stage", "").isEmpty()
                                ? "" : " (" + event.optString("stage") + ")"));
                break;
            default:
                break;
        }
        scrollDown();
    }

    /** Renders the transcript of an existing session (from a history reply). */
    public void renderHistory(JSONArray turns) {
        clear();
        if (turns == null) {
            return;
        }
        for (int index = 0; index < turns.length(); index++) {
            final JSONObject turn = turns.optJSONObject(index);
            if (turn == null) {
                continue;
            }
            beginTurn(turn.optString("user", ""), turn.optBoolean("has_image", false),
                    turn.optString("image_note", ""));
            final JSONArray recalled = turn.optJSONArray("recalled");
            if (recalled != null && recalled.length() > 0) {
                final JSONObject recallEvent = new JSONObject();
                try {
                    recallEvent.put("items", recalled);
                    recallEvent.put("count", recalled.length());
                } catch (Exception ignored) {
                }
                handleEvent(recallEvent);
            }
            final JSONArray plan = turn.optJSONArray("plan");
            if (plan != null && plan.length() > 0) {
                final JSONObject planEvent = new JSONObject();
                try {
                    planEvent.put("steps", plan);
                    planEvent.put("source", "final");
                } catch (Exception ignored) {
                }
                handleEvent(planEvent);
            }
            final JSONArray calls = turn.optJSONArray("tool_calls");
            for (int call = 0; calls != null && call < calls.length(); call++) {
                final JSONObject entry = calls.optJSONObject(call);
                if (entry == null) {
                    continue;
                }
                try {
                    final JSONObject callEvent = new JSONObject();
                    callEvent.put("name", entry.optString("name", ""));
                    callEvent.put("arguments", entry.optJSONObject("arguments") == null
                            ? new JSONObject() : entry.optJSONObject("arguments"));
                    handleEvent(callEvent.put("type", "tool_call"));
                    final JSONObject resultEvent = new JSONObject();
                    resultEvent.put("name", entry.optString("name", ""));
                    resultEvent.put("ok", entry.optBoolean("ok", false));
                    resultEvent.put("summary", entry.optString("summary", ""));
                    resultEvent.put("error", entry.optString("error", ""));
                    resultEvent.put("duration_ms", entry.optDouble("duration_ms", 0.0));
                    handleEvent(resultEvent.put("type", "tool_result"));
                } catch (Exception ignored) {
                }
            }
            if (assistantBubble != null) {
                assistantBubble.setText(turn.optString("answer", ""));
            }
            if (metricsLine != null) {
                metricsLine.setText(String.format(Locale.US,
                        "TTFT %.0f ms · prompt %d · gen %d · stop %s",
                        turn.optDouble("ttft_ms", 0.0),
                        turn.optLong("prompt_tokens", 0),
                        turn.optLong("generated_tokens", 0),
                        turn.optString("stop_reason", "-")));
            }
            turnRunning = false;
        }
        scrollDown();
    }

    // --- rendering helpers -------------------------------------------------

    private void renderPlan(JSONObject event) {
        final JSONArray steps = event.optJSONArray("steps");
        if (steps == null || steps.length() == 0) {
            return;
        }
        if (planCard == null) {
            final LinearLayout card = card("计划 Plan");
            planBody = (TextView) card.getChildAt(1);
            ensureSteps().addView(card);
            planCard = card;
        }
        final StringBuilder text = new StringBuilder();
        for (int index = 0; index < steps.length(); index++) {
            final JSONObject step = steps.optJSONObject(index);
            if (step == null) {
                continue;
            }
            final String status = step.optString("status", "pending");
            text.append(statusMark(status)).append(' ')
                    .append(step.optInt("index", index + 1)).append(". ")
                    .append(step.optString("text", ""));
            if (step.optString("note", "").length() > 0) {
                text.append(" — ").append(step.optString("note"));
            }
            text.append('\n');
        }
        planBody.setText(text.toString().trim());
    }

    private void appendDelta(JSONObject event) {
        if (liveStep == null) {
            liveCard = card("模型输出中…");
            liveStep = (TextView) liveCard.getChildAt(1);
            ensureSteps().addView(liveCard);
        }
        final String piece;
        try {
            piece = decoder.decode(Base64.decode(event.optString("text_b64", ""),
                    Base64.DEFAULT));
        } catch (IllegalArgumentException error) {
            return;
        }
        liveStep.append(piece);
    }

    private void finishStep(JSONObject event) {
        final String kind = event.optString("kind", "");
        final String text = event.optString("text", "");
        if ("answer".equals(kind)) {
            if (liveCard != null && currentSteps != null) {
                currentSteps.removeView(liveCard);
                liveCard = null;
                liveStep = null;
            }
            if (assistantBubble != null && text.length() > 0) {
                assistantBubble.setText(text);
            }
            return;
        }
        // A tool step: keep whatever text the model produced, it explains the call.
        if (liveStep != null) {
            final String cleaned = text.replaceAll("<\\|tool_call>|<tool_call\\|>", "").trim();
            liveStep.setText(cleaned.isEmpty() ? "(tool call)" : cleaned);
        }
    }

    private void openToolCard(JSONObject event) {
        final String name = event.optString("name", "?");
        final LinearLayout card = card("工具调用 · " + name);
        final TextView body = (TextView) card.getChildAt(1);
        body.setText(event.optJSONObject("arguments") == null
                ? "{}" : event.optJSONObject("arguments").toString());
        ensureSteps().addView(card);
        lastToolCard = card;
    }

    private void closeToolCard(JSONObject event) {
        if (lastToolCard == null) {
            return;
        }
        final TextView body = (TextView) lastToolCard.getChildAt(1);
        final boolean ok = event.optBoolean("ok", false);
        final String detail = ok
                ? event.optString("summary", "")
                : "error: " + event.optString("error", "");
        body.append("\n→ " + (ok ? "✓ " : "✗ ") + detail);
        final double duration = event.optDouble("duration_ms", 0.0);
        if (duration > 0.0) {
            body.append(String.format(Locale.US, "  (%.1f ms)", duration));
        }
        final View title = lastToolCard.getChildAt(0);
        if (title instanceof TextView) {
            ((TextView) title).setTextColor(ok ? COLOR_ACCENT : COLOR_WARN);
        }
        lastToolCard = null;
    }

    private void addStep(String title, String body, int color) {
        final LinearLayout card = card(title);
        ((TextView) card.getChildAt(0)).setTextColor(color);
        ((TextView) card.getChildAt(1)).setText(body);
        ensureSteps().addView(card);
    }

    /** "更新记忆" when a slot was replaced, "合并记忆" for a duplicate. */
    private String memoryActionLabel(JSONObject event) {
        final String action = event.optString("action", "");
        if ("updated".equals(action)) {
            return "更新记忆";
        }
        if (event.optBoolean("merged", false)) {
            return "合并记忆";
        }
        return "写入记忆";
    }

    private String memoryActionNote(JSONObject event) {
        final String action = event.optString("action", "");
        if ("updated".equals(action)) {
            return "  (原先的说法已被替换)";
        }
        if (event.optBoolean("merged", false)) {
            return "  (与已有记忆相同，已合并)";
        }
        return "";
    }

    private LinearLayout card(String title) {
        final LinearLayout card = new LinearLayout(context);
        card.setOrientation(LinearLayout.VERTICAL);
        final GradientDrawable background = rounded(COLOR_CARD, 10);
        background.setStroke(dp(1), COLOR_CARD_BORDER);
        card.setBackground(background);
        card.setPadding(dp(10), dp(8), dp(10), dp(8));
        final LinearLayout.LayoutParams params = new LinearLayout.LayoutParams(
                ViewGroup.LayoutParams.MATCH_PARENT, ViewGroup.LayoutParams.WRAP_CONTENT);
        params.topMargin = dp(6);
        params.rightMargin = dp(40);
        card.setLayoutParams(params);

        final TextView header = label(title, 11, COLOR_ACCENT, true);
        card.addView(header);
        final TextView body = label("", 12, COLOR_TEXT, true);
        body.setTextIsSelectable(true);
        card.addView(body);
        return card;
    }

    /**
     * The container for step cards.  Created on demand so an event that arrives
     * without an open turn (replayed log, another client) can never dereference
     * a null view.
     */
    private LinearLayout ensureSteps() {
        if (currentSteps == null) {
            currentSteps = new LinearLayout(context);
            currentSteps.setOrientation(LinearLayout.VERTICAL);
            if (currentTurn != null) {
                currentTurn.addView(currentSteps);
            } else {
                container.addView(currentSteps);
            }
        }
        return currentSteps;
    }

    private TextView label(String text, int sizeSp, int color, boolean monospace) {
        final TextView view = new TextView(context);
        view.setText(text);
        view.setTextSize(TypedValue.COMPLEX_UNIT_SP, sizeSp);
        view.setTextColor(color);
        if (monospace) {
            view.setTypeface(Typeface.MONOSPACE);
        }
        view.setLineSpacing(dp(2), 1.0f);
        Linkify.addLinks(view, Linkify.WEB_URLS);
        return view;
    }

    private GradientDrawable rounded(int color, int radiusDp) {
        final GradientDrawable drawable = new GradientDrawable();
        drawable.setColor(color);
        drawable.setCornerRadius(dp(radiusDp));
        return drawable;
    }

    private String statusMark(String status) {
        switch (status) {
            case "done":
                return "✔";
            case "active":
                return "▶";
            case "failed":
                return "✗";
            case "skipped":
                return "–";
            default:
                return "○";
        }
    }

    private void scrollDown() {
        if (!autoScroll) {
            return;
        }
        scroll.post(() -> scroll.fullScroll(View.FOCUS_DOWN));
    }

    private int dp(int value) {
        return Math.round(value * density);
    }

    private static int optLength(JSONObject object, String key) {
        final JSONArray array = object.optJSONArray(key);
        return array == null ? 0 : array.length();
    }
}
