package io.github.lxw112190.ppocr.benchmark;

import io.github.lxw112190.ppocr.ppocr.ParallelismPlan;

/** Emits the AUTO plan for the processor count visible to the JVM. */
public final class ParallelismPolicyMain {
    private ParallelismPolicyMain() { }

    public static void main(String[] args) {
        int lines = args.length > 0 ? Integer.parseInt(args[0]) : 16;
        int processors = Math.max(1, Runtime.getRuntime().availableProcessors());
        ParallelismPlan plan = ParallelismPlan.automatic(processors, lines);
        System.out.println("{\"benchmark\":\"parallelism-policy\",\"mode\":\"auto\","
                + "\"available_processors\":" + plan.getAvailableProcessors()
                + ",\"lines\":" + lines
                + ",\"line_workers\":" + plan.getLineWorkers()
                + ",\"classifier_workers\":" + plan.getClassifierWorkers()
                + ",\"recognizer_workers\":" + plan.getRecognizerWorkers()
                + ",\"recognizer_intra_op\":" + plan.getRecognizerIntraOp()
                + ",\"detector_intra_op\":" + plan.getDetectorIntraOp()
                + ",\"rec_cpu_budget\":"
                + plan.getLineWorkers() * plan.getRecognizerIntraOp() + "}");
    }
}
