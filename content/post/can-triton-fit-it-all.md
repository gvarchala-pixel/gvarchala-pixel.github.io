---
title: "Can Triton Fit It All?"
date: 2026-07-05T09:00:00-06:00
subtitle: "One server, multiple frameworks, zero apologies for the simplification."
layout: post
tags: ["ML", "MLOps", "systems", "production", "model-serving"]
categories: ["general"]
description: "An intro to NVIDIA Triton Inference Server, why decoupling model inference from the serving flow solves a real resource and versioning problem, and where Triton stops making sense."
---

The serving flow I was working on had gotten tangled in a fairly ordinary way. A single request needed features pulled from a feature store, some features computed from the user context, and a few more built directly from the incoming request. All of that fed into a model, and the model produced the prediction that actually mattered. The problem was that all of it, the feature assembly and the model inference, lived inside the same service.

That coupling created a resource problem before it created anything else. The model was the part that actually needed a GPU and careful capacity planning, but because everything ran in one service, scaling for the model meant scaling the feature assembly logic, the Kafka logging that fed business metrics and reporting, and a reproducibility pipeline that sampled about one percent of feature vectors for later auditing, whether any of that needed the extra resources or not. Every capacity decision became a decision about the whole service, even when only one piece of it was under real pressure.

A/B testing suffered for the same reason. Shipping a new model version meant touching the same deployment that also handled feature orchestration, so a model change and a serving change were never really separate events even when they should have been.

Pulling model loading and inference out onto Triton fixed the coupling directly. The serving flow kept doing what it was good at, assembling features from the store, the request, and the user context. Triton took over the one job it is actually built for, loading and running the model. Resource allocation could finally follow the real bottleneck instead of the whole service. Rolling out a new model version became a change inside Triton instead of a redeploy of everything around it, which is what made A/B testing genuinely painless for the first time.

There is an auditability benefit that only shows up once a system has been running long enough to get complicated. Sampling a slice of feature vectors for reproducibility only means something if you can also tie each prediction back to the exact model version that produced it, and that link is much cleaner when the model lives in a place built to track versions in the first place. With more features and tighter SLAs stacking up over time, having one place that owns model versioning is worth more than it sounds like on paper.

None of this is free. Decoupling means an extra network hop between the serving layer and Triton, and a new piece of infrastructure to operate and monitor. It was worth it in this case, but it is a tradeoff rather than a default answer, and it is worth weighing honestly for any system considering the same move.

Triton Inference Server, officially renamed NVIDIA Dynamo Triton after a March 2025 rebrand though most engineers still just call it Triton, is what made this possible.

---

## So What Is Triton, Actually

Triton is a server, though calling it that undersells the job a little. Most servers hand you data. Triton hands you a decision, because the whole point of running a request through it is to get a model's opinion back.

The trick is in how it forms that opinion. Triton does not care whether the model was trained in PyTorch, exported to ONNX, or compiled down to TensorRT for a GPU. Each of these gets handled by something Triton calls a backend, and a backend is really just a small adapter that knows how to load and run one particular kind of model. Think of it as a universal power adapter for models trained on completely different plugs. When a new model type shows up, someone writes a new backend for it, and Triton learns the new trick without anyone touching the rest of the system.

Because every model sits behind the same backend interface, one Triton instance can hold a PyTorch model, an ONNX model, and a TensorRT model at once, in the same process, on the same machine, without any of them knowing the others exist. A client sends a request over HTTP or gRPC, names the model it wants, and Triton routes the request to the right backend. The client never has to know or care what framework is doing the actual thinking.

A few things come along for free once this is set up. Triton can batch requests from multiple clients together before sending them to the GPU, which is the difference between a bus running full and a taxi making five separate trips. It can hold several versions of the same model at once, so a new version rolls out gradually while the old one keeps working the night shift. And it exposes health checks and metrics out of the box, so nobody has to reinvent that wheel for the fifth time this year, in a slightly different shape, with a slightly different bug.

---

## Before and After, Drawn Out

The resource problem is easier to see than to explain, so here is the shape of it both ways.

![One service handling feature assembly, logging, and model inference together](/img/triton/before-coupled.svg)
*Before: the model competes for GPU and capacity decisions with everything else living in the same deployment.*

![Serving layer and Triton as two separate deployments connected over gRPC](/img/triton/after-decoupled.svg)
*After: the model gets its own resource pool and its own version history, and a new release ships without touching the feature logic around it.*

---

## Who's Actually Running This Thing

An intro post is easy to write in a vacuum. The harder and more honest question is whether real companies run this in production, or whether it just sounds good in a slide deck.

Public documentation and case studies suggest a fairly consistent pattern. NVIDIA claims over 25,000 companies deploy its AI inference stack in some form, which is a big enough number to be almost meaningless on its own, but the named examples are more useful. Microsoft runs Triton inside Azure Cognitive Services to generate live transcriptions during Teams calls, and also offers it as a managed inference endpoint inside Azure Machine Learning. Snap uses it for real time serving and reports cutting serving costs by close to half while doubling their latency performance. Samsung Medison runs it for medical imaging models, where speed and accuracy both carry real consequences. Siemens Energy uses it to bring machine learning to power plants that are full of sensors and cameras but were built on infrastructure from a much older era.

The pattern across all of these is not that Triton is the only option. TensorFlow Serving stays the tighter choice when every model in an organization happens to be TensorFlow, since it skips multi framework overhead entirely by design. TorchServe, built by AWS and Meta, shows up most often on teams that iterate quickly and want custom Python handlers without friction. Triton's documentation and its adopters both point to the same two reasons for choosing it, model diversity across frameworks and GPU utilization that matters enough to actively engineer around. KServe is worth a mention on its own terms too, since based on how it describes itself, it does not compete with Triton so much as sit in front of it, letting Kubernetes route requests to whichever serving backend a team has chosen underneath.

So the short answer is yes, real companies run this in production, and not just the ones with a marketing reason to say so.

---

## Where Triton Stops Making Sense

Every decoupling decision trades one kind of complexity for another. Running Triton means running one more service, one more thing to monitor, and one more hop on the request path between the serving layer and the model. For a team running two or three models, that overhead is real and probably not worth it. A single well written serving container will do the job with far less operational weight.

The case for Triton gets stronger as the model count grows, or as GPU cost starts mattering enough to actively engineer around, or as a platform team ends up owning inference for several other teams at once. At that point the alternative is not "no serving layer," it is five or ten teams each maintaining their own smaller version of the same problem, and that tends to be the more expensive path even though it looks cheaper up front.

The honest answer to whether this is over engineering depends entirely on where a team sits on that curve, and it is worth asking the question before adopting Triton rather than after.

---

## References

The following sources were used to verify claims made in this post, and the illustrations were generated with Claude.

- **NVIDIA Triton documentation**: `docs.nvidia.com/deeplearning/triton-inference-server`
- **Triton GitHub repository**: `github.com/triton-inference-server/server`
- **NVIDIA Newsroom, 25,000+ companies deploying NVIDIA AI inference**: `nvidianews.nvidia.com`
- **NVIDIA Dynamo Triton rebrand, announced at GTC March 2025**: `nvidia.com/en-us/ai/dynamo-triton`
- **KServe frameworks overview**: `kserve.github.io/website/docs/model-serving/predictive-inference/frameworks/overview`
- **Illustrations**: generated with Claude for this post
