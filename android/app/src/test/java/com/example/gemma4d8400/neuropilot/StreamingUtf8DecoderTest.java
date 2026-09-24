package com.example.gemma4d8400.neuropilot;

import java.nio.charset.StandardCharsets;
import java.util.Base64;

public final class StreamingUtf8DecoderTest {
    public static void main(String[] args) {
        StreamingUtf8Decoder decoder = new StreamingUtf8Decoder();
        String first = decoder.decode(Base64.getDecoder().decode("6Zs="));
        String second = decoder.decode(Base64.getDecoder().decode("iQ=="));
        String tail = decoder.finish();
        if (!first.isEmpty() || !"\u96c9".equals(second) || !tail.isEmpty()) {
            throw new AssertionError(
                    "split UTF-8 decode failed: first=" + first
                            + ", second=" + second + ", tail=" + tail);
        }

        decoder.reset();
        String complete = decoder.decode("\u96c9\u9e21".getBytes(StandardCharsets.UTF_8));
        if (!"\u96c9\u9e21".equals(complete) || !decoder.finish().isEmpty()) {
            throw new AssertionError("complete UTF-8 decode failed: " + complete);
        }
    }
}
