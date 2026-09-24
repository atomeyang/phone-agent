package com.example.gemma4d8400.neuropilot;

import android.util.Log;

import org.json.JSONException;
import org.json.JSONObject;

import java.io.File;
import java.io.ByteArrayOutputStream;
import java.io.FileOutputStream;
import java.io.IOException;
import java.io.RandomAccessFile;
import java.nio.charset.StandardCharsets;

/**
 * File based bridge to the native agent.
 *
 * <p>The runner owns every piece of agent logic (memory, planning, tools,
 * history) and publishes an append-only event log.  This class only writes
 * request documents and tails that log, so the app never duplicates agent
 * behaviour - it renders what native code decided.
 */
public final class AgentClient {

    public interface EventListener {
        /** Called on the polling thread for every complete event line. */
        void onEvent(JSONObject event);

        /** Called on the polling thread when the event log restarted. */
        void onLogReset();
    }

    private static final String TAG = "Gemma4Agent";
    private static final long POLL_INTERVAL_MS = 120L;
    private static final long RESPONSE_TIMEOUT_MS = 120_000L;

    private final File agentDir;
    private final File requestFile;
    private final File responseFile;
    private final File eventsFile;
    private final File cancelFile;
    private final File readyFile;

    private volatile boolean polling;
    private Thread pollThread;
    private long lastReadyStamp;

    public AgentClient(File runtimeRoot) {
        this.agentDir = new File(runtimeRoot, "agent");
        this.requestFile = new File(agentDir, "request.json");
        this.responseFile = new File(agentDir, "response.json");
        this.eventsFile = new File(agentDir, "events.jsonl");
        this.cancelFile = new File(agentDir, "cancel");
        this.readyFile = new File(agentDir, "service-ready");
    }

    public File agentDirectory() {
        return agentDir;
    }

    public File eventsFile() {
        return eventsFile;
    }

    public boolean isReady() {
        return readyFile.isFile();
    }

    public long readyStamp() {
        return readyFile.isFile() ? readyFile.lastModified() : 0L;
    }

    /** Fire and forget: used for turns, which answer through the event log. */
    public void send(JSONObject request) throws IOException {
        writeAtomic(requestFile, request.toString());
    }

    /**
     * Sends a request and waits for the runner to publish response.json.  Used
     * for the read-only queries the UI needs (history, memory list, sessions).
     */
    public JSONObject call(JSONObject request) throws IOException, JSONException {
        // Remove the previous answer first.  Comparing timestamps was not enough:
        // the runner publishes response.json per request, so a reader that
        // arrives just after a write saw `lastModified() >= before` on its first
        // poll and returned the *previous* document (the app opened a new session
        // and showed an empty id because it parsed the old history dump).
        if (responseFile.isFile() && !responseFile.delete()) {
            Log.w(TAG, "cannot clear the previous response document");
        }
        send(request);
        final long deadline = System.currentTimeMillis() + RESPONSE_TIMEOUT_MS;
        while (System.currentTimeMillis() < deadline) {
            if (responseFile.isFile()) {
                final String text = readText(responseFile);
                if (!text.isEmpty()) {
                    return new JSONObject(text);
                }
            }
            try {
                Thread.sleep(25L);
            } catch (InterruptedException interrupted) {
                Thread.currentThread().interrupt();
                break;
            }
        }
        throw new IOException("the agent did not answer in time");
    }

    public void cancelCurrentTurn() {
        try (FileOutputStream ignored = new FileOutputStream(cancelFile)) {
        } catch (IOException error) {
            Log.w(TAG, "cannot write the cancel sentinel", error);
        }
    }

    public void startPolling(EventListener listener) {
        stopPolling();
        polling = true;
        pollThread = new Thread(() -> pollLoop(listener), "agent-events");
        pollThread.setDaemon(true);
        pollThread.start();
    }

    public void stopPolling() {
        polling = false;
        if (pollThread != null) {
            pollThread.interrupt();
            try {
                pollThread.join(500L);
            } catch (InterruptedException interrupted) {
                Thread.currentThread().interrupt();
            }
            pollThread = null;
        }
    }

    private void pollLoop(EventListener listener) {
        // Follow the log from its current end: completed turns are rendered from
        // the `history` request, so replaying the backlog here would draw every
        // past turn a second time (and used to crash the renderer).
        long offset = eventsFile.isFile() ? eventsFile.length() : 0L;
        long knownReadyStamp = readyStamp();
        // RandomAccessFile.readLine() decodes bytes as ISO-8859-1, which turns
        // every non-ASCII character of the event log (all the Chinese card text
        // and answers) into mojibake.  Read bytes, cut complete lines and decode
        // those as UTF-8; a trailing partial line is kept for the next poll so a
        // multi-byte character split across reads is still decoded correctly.
        final ByteArrayOutputStream pending = new ByteArrayOutputStream();
        while (polling) {
            try {
                // A restarted runner truncates the log: replay from the start.
                final long currentReadyStamp = readyStamp();
                if (currentReadyStamp != knownReadyStamp) {
                    knownReadyStamp = currentReadyStamp;
                    offset = 0L;
                    pending.reset();
                    listener.onLogReset();
                }
                if (!eventsFile.isFile()) {
                    Thread.sleep(POLL_INTERVAL_MS);
                    continue;
                }
                final long length = eventsFile.length();
                if (length < offset) {
                    offset = 0L;   // the file was replaced under us
                    pending.reset();
                }
                if (length == offset) {
                    Thread.sleep(POLL_INTERVAL_MS);
                    continue;
                }
                try (RandomAccessFile reader = new RandomAccessFile(eventsFile, "r")) {
                    reader.seek(offset);
                    final byte[] chunk = new byte[64 * 1024];
                    int count;
                    while ((count = reader.read(chunk)) > 0) {
                        for (int index = 0; index < count; index++) {
                            final byte value = chunk[index];
                            if (value == '\n') {
                                final String line = new String(pending.toByteArray(),
                                        StandardCharsets.UTF_8).trim();
                                pending.reset();
                                final JSONObject event = parse(line);
                                if (event != null) {
                                    listener.onEvent(event);
                                }
                            } else if (value != '\r') {
                                pending.write(value);
                            }
                        }
                        offset = reader.getFilePointer();
                    }
                }
            } catch (InterruptedException interrupted) {
                Thread.currentThread().interrupt();
                return;
            } catch (IOException error) {
                Log.w(TAG, "event log read failed", error);
            }
            try {
                Thread.sleep(POLL_INTERVAL_MS);
            } catch (InterruptedException interrupted) {
                Thread.currentThread().interrupt();
                return;
            }
        }
    }

    private static JSONObject parse(String line) {
        final String trimmed = line == null ? "" : line.trim();
        if (trimmed.isEmpty()) {
            return null;
        }
        try {
            return new JSONObject(trimmed);
        } catch (JSONException error) {
            Log.w(TAG, "bad event line: " + trimmed, error);
            return null;
        }
    }

    private static void writeAtomic(File target, String contents) throws IOException {
        final File parent = target.getParentFile();
        if (parent != null && !parent.isDirectory() && !parent.mkdirs()) {
            throw new IOException("cannot create " + parent);
        }
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

    private static String readText(File file) throws IOException {
        try (RandomAccessFile reader = new RandomAccessFile(file, "r")) {
            final byte[] buffer = new byte[(int) reader.length()];
            reader.readFully(buffer);
            return new String(buffer, StandardCharsets.UTF_8);
        }
    }
}
