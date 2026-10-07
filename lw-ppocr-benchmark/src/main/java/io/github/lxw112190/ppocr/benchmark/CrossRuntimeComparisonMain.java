package io.github.lxw112190.ppocr.benchmark;
import io.github.lxw112190.ppocr.image.BgrImage;
import io.github.lxw112190.ppocr.imageio.ImageIoLoader;
import io.github.lxw112190.ppocr.kernels.KernelBackend;
import io.github.lxw112190.ppocr.ppocr.*;
import java.nio.file.*;
import java.security.MessageDigest;
import java.util.*;
import java.lang.management.*;
import java.io.PrintStream;

/** Explicit same-pixel CPU comparison; independent of the CI performance gate. */
public final class CrossRuntimeComparisonMain {
    private CrossRuntimeComparisonMain() { }
    public static void main(String[] args) throws Exception {
        // Native redirected stdout may otherwise use the Windows code page.
        System.setOut(new PrintStream(System.out,true,"UTF-8"));
        if(args.length==3 && args[0].equals("export-image")) {
            BgrImage image=ImageIoLoader.load(Paths.get(args[1]));
            byte[] pixels=new byte[Math.multiplyExact(Math.multiplyExact(image.width(),image.height()),3)];
            for(int y=0;y<image.height();y++) System.arraycopy(image.pixels(),image.offset()+y*image.stride(),
                pixels,y*image.width()*3,image.width()*3);
            Files.write(Paths.get(args[2]),pixels);
            System.out.printf(Locale.ROOT,"{\"width\":%d,\"height\":%d,\"stride\":%d,\"bgr_sha256\":%s}%n",
                image.width(),image.height(),image.width()*3,quote(hash(pixels)));
            return;
        }
        if(args.length!=8 || !args[0].equals("run")) throw new IllegalArgumentException(
            "export-image image output.bgr | run models variant input.bgr width height warmup iterations");
        Path root=Paths.get(args[1]); String variant=args[2];
        if(!Arrays.asList("tiny","small","medium").contains(variant)) throw new IllegalArgumentException("variant");
        int w=positive(args[4]),h=positive(args[5]),warmup=positive(args[6]),iterations=positive(args[7]);
        byte[] pixels=Files.readAllBytes(Paths.get(args[3]));
        if(pixels.length!=(long)w*h*3) throw new IllegalArgumentException("packed BGR size mismatch");
        BgrImage image=new BgrImage(pixels,w,h,w*3);
        KernelBackend backend=(KernelBackend)Class.forName("io.github.lxw112190.ppocr.vector.VectorBackend")
            .getDeclaredConstructor().newInstance();
        PaddleOcrOptions options=PaddleOcrOptions.builder().setParallelismMode(ParallelismPolicy.AUTO)
            .setDetectionMaximumSideLength(960).setDetectionBitmapThreshold(.3f)
            .setDetectionBoxThreshold(.6f).setDetectionUnclipRatio(1.6f)
            .setDetectionDilation(false).setMaxDetectionCandidates(1000).setClassifierThreshold(.9f).build();
        Path directory=root.resolve("ppocrv6-"+variant);
        try(PaddleOcr ocr=PaddleOcr.load(directory.resolve("det.onnx"),root.resolve("ppocrv6-tiny/cls.onnx"),
            directory.resolve("rec.onnx"),root.resolve(variant.equals("tiny")?"ppocrv6-tiny/ppocr_keys.txt":
            "ppocrv6-shared/PP-OCRv6_small_rec_dict.txt"),options,backend)) {
            OcrResult result=null;
            for(int i=0;i<warmup;i++) result=ocr.recognize(image);
            String reference=result.getText(); int count=result.getLines().size();
            long[] samples=new long[iterations]; long gcBefore=gc();
            for(int i=0;i<iterations;i++) {
                long start=System.nanoTime(); result=ocr.recognize(image); samples[i]=System.nanoTime()-start;
                if(result.getLines().size()!=count || !result.getText().equals(reference))
                    throw new IllegalStateException("unstable OCR result");
            }
            long gcDelta=gc()-gcBefore; Arrays.sort(samples);
            double total=0; for(long sample:samples) total+=sample;
            System.gc(); long heap=ManagementFactory.getMemoryMXBean().getHeapMemoryUsage().getUsed();
            ParallelismPlan plan=ParallelismPlan.automatic(Runtime.getRuntime().availableProcessors(),count);
            DetInputShape detShape=DetInputShapePolicy.choose(image,options.getDetectionMaximumSideLength());
            StringBuilder lines=new StringBuilder("[");
            for(OcrLineResult line:result.getLines()) {
                if(lines.length()>1) lines.append(',');
                BgrImage crop=PerspectiveCrop.crop(image,line.getBox());
                lines.append("{\"text\":").append(quote(line.getText())).append(",\"rotation\":")
                    .append(line.isRotated()?180:0).append(",\"rec_width\":")
                    .append(RecWidthPolicy.chooseTargetWidth(crop,960)).append(",\"box\":")
                    .append(Arrays.toString(line.getBox().getPoints())).append(",\"score\":")
                    .append(line.getRecognitionScore()).append('}');
            }
            System.out.printf(Locale.ROOT,"{\"schema\":1,\"runtime\":\"java\",\"variant\":%s,"
                +"\"fma\":%s,\"cpu\":%d,\"line_workers\":%d,\"det_threads\":%d,\"rec_threads\":%d,"
                +"\"warmup\":%d,\"iterations\":%d,\"width\":%d,\"height\":%d,\"det_width\":%d,\"det_height\":%d,\"bgr_sha256\":%s,"
                +"\"settings\":{\"det_limit\":960,\"bitmap\":0.3,\"box\":0.6,\"unclip\":1.6,\"dilation\":false,\"max_candidates\":1000,\"cls_threshold\":0.9,\"rec_buckets\":[192,320,480,640,960]},"
                +"\"mean_ms\":%.3f,\"median_ms\":%.3f,\"p95_ms\":%.3f,\"heap_after_gc_bytes\":%d,"
                +"\"gc_count\":%d,\"lines\":%s}%n",quote(variant),Boolean.getBoolean("lwppocr.vectorFma"),
                Runtime.getRuntime().availableProcessors(),plan.getLineWorkers(),plan.getDetectorIntraOp(),
                plan.getRecognizerIntraOp(),warmup,iterations,w,h,detShape.getInputWidth(),detShape.getInputHeight(),
                quote(hash(pixels)),total/iterations/1e6,
                samples[samples.length/2]/1e6,samples[(int)Math.ceil(iterations*.95)-1]/1e6,heap,gcDelta,
                lines.append(']').toString());
        }
    }
    private static int positive(String s) { int n=Integer.parseInt(s); if(n<=0) throw new IllegalArgumentException("positive argument required"); return n; }
    private static long gc() { long n=0; for(GarbageCollectorMXBean b:ManagementFactory.getGarbageCollectorMXBeans()) if(b.getCollectionCount()>0)n+=b.getCollectionCount(); return n; }
    private static String hash(byte[] bytes) throws Exception { StringBuilder s=new StringBuilder(); for(byte b:MessageDigest.getInstance("SHA-256").digest(bytes))s.append(String.format(Locale.ROOT,"%02x",b&255));return s.toString(); }
    static String quote(String s) {
        StringBuilder b=new StringBuilder("\"");
        for(int i=0;i<s.length();i++) { char c=s.charAt(i); if(c=='"' || c=='\\') b.append('\\').append(c);
            else if(c<32)b.append(String.format(Locale.ROOT,"\\u%04x",(int)c)); else b.append(c); }
        return b.append('"').toString();
    }
}
