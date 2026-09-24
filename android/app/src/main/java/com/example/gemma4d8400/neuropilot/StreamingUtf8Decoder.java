package com.example.gemma4d8400.neuropilot;

import java.nio.ByteBuffer;
import java.nio.CharBuffer;
import java.nio.charset.CharacterCodingException;
import java.nio.charset.CharsetDecoder;
import java.nio.charset.CodingErrorAction;
import java.nio.charset.CoderResult;
import java.nio.charset.StandardCharsets;

final class StreamingUtf8Decoder {
    private final CharsetDecoder decoder = StandardCharsets.UTF_8.newDecoder()
            .onMalformedInput(CodingErrorAction.REPLACE)
            .onUnmappableCharacter(CodingErrorAction.REPLACE);
    private ByteBuffer pending = ByteBuffer.allocate(16);

    void reset() {
        decoder.reset();
        pending.clear();
    }

    String decode(byte[] bytes) {
        ensureCapacity(bytes.length);
        pending.put(bytes);
        return drain(false);
    }

    String finish() {
        return drain(true);
    }

    private String drain(boolean endOfInput) {
        pending.flip();
        CharBuffer output = CharBuffer.allocate(Math.max(8, pending.remaining() * 2 + 8));
        try {
            CoderResult result = decoder.decode(pending, output, endOfInput);
            throwIfError(result);
            if (endOfInput) {
                throwIfError(decoder.flush(output));
                pending.clear();
                decoder.reset();
            } else {
                pending.compact();
            }
        } catch (CharacterCodingException error) {
            reset();
            return "\ufffd";
        }
        output.flip();
        return output.toString();
    }

    private void ensureCapacity(int additionalBytes) {
        if (pending.remaining() >= additionalBytes) {
            return;
        }
        int required = pending.position() + additionalBytes;
        int capacity = Math.max(pending.capacity() * 2, required);
        ByteBuffer expanded = ByteBuffer.allocate(capacity);
        pending.flip();
        expanded.put(pending);
        pending = expanded;
    }

    private static void throwIfError(CoderResult result)
            throws CharacterCodingException {
        if (result.isError()) {
            result.throwException();
        }
    }
}
