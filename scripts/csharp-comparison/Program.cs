using Sdcb.SimdPaddleOCR;
using System.Diagnostics;
using System.Reflection;
using System.Security.Cryptography;
using System.Text.Json;
using System.Runtime.InteropServices;
using System.Runtime.Intrinsics.X86;

if(args.Length!=7) throw new ArgumentException("models variant input.bgr width height warmup iterations");
string root=args[0],variant=args[1];
if(variant is not ("tiny" or "small" or "medium")) throw new ArgumentException("variant");
int w=int.Parse(args[3]),h=int.Parse(args[4]),warmup=int.Parse(args[5]),iterations=int.Parse(args[6]);
if(w<=0||h<=0||warmup<=0||iterations<=0) throw new ArgumentException("positive arguments required");
byte[] bgr=File.ReadAllBytes(args[2]);
if(bgr.LongLength!=(long)w*h*3) throw new ArgumentException("packed BGR size mismatch");
var rec=new PaddleOcrRecognizerOptions { AdaptiveWidth=true,TargetWidth=960 };
var hook=typeof(PaddleOcrRecognizerOptions).GetProperty("UseLwAdaptiveWidth",BindingFlags.NonPublic|BindingFlags.Instance)
    ?? throw new InvalidOperationException("Pinned C# comparison width hook is missing");
hook.SetValue(rec,true);
if(hook.GetValue(rec) is not true) throw new InvalidOperationException("REC width hook not enabled");
int workers=Math.Min(Environment.ProcessorCount,4);
var options=new PaddleOcrOptions { LineWorkerCount=workers,DetIntraOpThreads=Math.Min(Environment.ProcessorCount,4),
    ClassifierThreshold=.9f,Recognizer=rec,
    Detector=new PaddleOcrDetectorOptions { LimitSideLength=960,MaxCandidates=1000,
        BitmapThreshold=.3f,BoxThreshold=.6f,UnclipRatio=1.6f,UseDilation=false } };
using var ocr=PaddleOcrAll.Load(Path.Combine(root,$"ppocrv6-{variant}/det.onnx"),
    Path.Combine(root,"ppocrv6-tiny/cls.onnx"),Path.Combine(root,$"ppocrv6-{variant}/rec.onnx"),
    Path.Combine(root,variant=="tiny"?"ppocrv6-tiny/ppocr_keys.txt":"ppocrv6-shared/PP-OCRv6_small_rec_dict.txt"),options);
PaddleOcrResult result=null!;
for(int i=0;i<warmup;i++)result=ocr.Run(bgr,w,h,w*3);
string reference=result.Text; int count=result.Lines.Length;
double[] samples=new double[iterations];
long gcBefore=Enumerable.Range(0,3).Sum(g=>GC.CollectionCount(g));
for(int i=0;i<iterations;i++) {
    long start=Stopwatch.GetTimestamp(); result=ocr.Run(bgr,w,h,w*3); samples[i]=Stopwatch.GetElapsedTime(start).TotalMilliseconds;
    if(result.Text!=reference||result.Lines.Length!=count)throw new InvalidOperationException("unstable OCR result");
}
long gcDelta=Enumerable.Range(0,3).Sum(g=>GC.CollectionCount(g))-gcBefore;
Array.Sort(samples);
long heap=GC.GetTotalMemory(true);
var assembly=typeof(PaddleOcrAll).Assembly;
var cropSize=assembly.GetType("Sdcb.SimdPaddleOCR.PPOCRCrop")!.GetMethod("GetSize",BindingFlags.Public|BindingFlags.Static)!;
var selectWidth=typeof(PaddleOcrRecognizer).GetMethod("SelectLwAdaptiveWidth",BindingFlags.NonPublic|BindingFlags.Static)!;
var lines=result.Lines.Select(l=>{
    var size=((int,int,int))cropSize.Invoke(null,new object[]{l.Box})!;
    int width=(int)selectWidth.Invoke(null,new object[]{size.Item1,size.Item2,960})!;
    return new {text=l.Text,rotation=l.AppliedRotationDegrees,rec_width=width,
        box=new[]{l.Box.X1,l.Box.Y1,l.Box.X2,l.Box.Y2,l.Box.X3,l.Box.Y3,l.Box.X4,l.Box.Y4},score=l.RecognitionScore};
}).ToArray();
Console.WriteLine(JsonSerializer.Serialize(new {
    schema=1,runtime="csharp",variant,fma=Fma.IsSupported,cpu=Environment.ProcessorCount,
    line_workers=ocr.EffectiveLineWorkerCount,det_threads=options.DetIntraOpThreads,
    rec_threads=Math.Clamp(Environment.ProcessorCount/workers,1,4),
    warmup,iterations,width=w,height=h,det_width=result.DetectorResizedWidth,det_height=result.DetectorResizedHeight,
    bgr_sha256=Convert.ToHexString(SHA256.HashData(bgr)).ToLowerInvariant(),
    settings=new {det_limit=960,bitmap=.3,box=.6,unclip=1.6,dilation=false,max_candidates=1000,
        cls_threshold=.9,rec_buckets=new[]{192,320,480,640,960}},
    mean_ms=samples.Average(),median_ms=samples[samples.Length/2],p95_ms=samples[(int)Math.Ceiling(iterations*.95)-1],
    heap_after_gc_bytes=heap,gc_count=gcDelta,lines,
    framework=RuntimeInformation.FrameworkDescription,avx2=Avx2.IsSupported,
    width_hook=true,fp32=true }));
