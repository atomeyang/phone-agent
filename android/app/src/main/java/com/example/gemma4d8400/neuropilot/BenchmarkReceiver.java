package com.example.gemma4d8400.neuropilot;

import android.content.BroadcastReceiver;
import android.content.Context;
import android.content.Intent;
import android.os.PowerManager;
import android.util.Log;

/**
 * Receives an external broadcast to trigger a screen-off or screen-on
 * benchmark without bringing the Activity to the foreground (which would
 * turn the screen on and raise frequencies).
 *
 * Trigger via:
 *   adb shell am broadcast -a com.example.gemma4d8400.neuropilot.RUN_BENCH \
 *       --ez screen_off true
 */
public final class BenchmarkReceiver extends BroadcastReceiver {
    static final String ACTION = "com.example.gemma4d8400.neuropilot.RUN_BENCH";
    static final String EXTRA_SCREEN_OFF = "screen_off";
    private static final String TAG = "PhoneVLM-BenchReceiver";

    @Override
    public void onReceive(Context context, Intent intent) {
        if (!ACTION.equals(intent.getAction())) {
            return;
        }
        boolean screenOff = intent.getBooleanExtra(EXTRA_SCREEN_OFF, true);
        Log.i(TAG, "received broadcast, screen_off=" + screenOff);

        // Acquire a partial wake lock so the process is not frozen by the
        // cgroup freezer while the screen is off.
        PowerManager pm = (PowerManager) context.getSystemService(Context.POWER_SERVICE);
        if (pm != null) {
            PowerManager.WakeLock wl = pm.newWakeLock(
                    PowerManager.PARTIAL_WAKE_LOCK, "PhoneVLM:bench-trigger");
            wl.acquire(300_000L); // 5 min max
            Log.i(TAG, "acquired PARTIAL_WAKE_LOCK");
        }

        // Forward to MainActivity via a foreground service intent.
        // We start the activity with FLAG_ACTIVITY_NEW_TASK so the receiver
        // has a valid context, but we do NOT turn the screen on — the
        // activity is already running (it was launched earlier). If the
        // process was frozen, the system will thaw it to deliver this
        // broadcast. We use goAsync to keep the receiver alive while the
        // benchmark runs on the worker thread.
        final PendingResult pendingResult = goAsync();
        new Thread(() -> {
            try {
                MainActivity.triggerBenchmarkFromReceiver(context, screenOff);
            } finally {
                pendingResult.finish();
            }
        }).start();
    }
}
