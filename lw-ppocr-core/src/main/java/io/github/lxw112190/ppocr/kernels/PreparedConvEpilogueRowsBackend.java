package io.github.lxw112190.ppocr.kernels;

/** Optional disjoint channel shards for prepared-convolution post-ops. */
public interface PreparedConvEpilogueRowsBackend {
    /** Channel boundaries must preserve flat activation Vector/tail boundaries. */
    int preparedEpilogueChannelAlignment();
    void finishPreparedConvChannels(float[] output,int offset,int channels,int plane,
            ConvEpilogue epilogue,float[] residual,int residualOffset,int firstChannel,int endChannel);
}
