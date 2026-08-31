# Journal target decision

## Primary target: Journal of Machine Learning Research (JMLR)

JMLR is the primary journal target for this work.

Why it fits:

- its scope explicitly includes theoretical studies that yield new insight into the design and behavior of learning systems;
- it explicitly includes new analytical frameworks that advance theoretical studies of learning methods;
- the paper's main contribution is a theoretical/analytical framework rather than a new benchmark architecture;
- JMLR encourages implementation and reproducibility material, which matches this repository's exact regression suite, frozen evidence programs, and figure/build automation;
- JMLR uses its public author-identified submission format, so the public repository can be supplied directly to reviewers.

Official author information: https://www.jmlr.org/author-info.html

Official formatting guide: https://www.jmlr.org/format/format.html

Official style repository: https://github.com/JmlrOrg/jmlr-style-file

The submission workspace is `paper/jmlr/` and the authoritative build is `.github/workflows/jmlr-build.yml`.

## TMLR: strong topical fit, not selected under the current project record

Transactions on Machine Learning Research is also a strong topical fit and welcomes theoretical/analytical frameworks. However, the current TMLR FAQ states an expectation that ideas, claims, and results be human-sourced when LLMs are used. This project used ChatGPT materially in conceptual/theoretical development as well as writing and implementation assistance. We therefore do not use TMLR as the primary target under the current research record, even though TMLR's broader editorial LLM policy permits assistive use with author responsibility.

Official FAQ: https://jmlr.org/tmlr/faq.html

Official editorial policies: https://jmlr.org/tmlr/editorial-policies.html

## Fallback: Neural Networks / related archival ML journals

Elsevier's *Neural Networks* welcomes mathematical and computational analyses of learning systems and Elsevier's current generative-AI policy permits disclosed AI assistance in research and manuscript preparation with human accountability. It is a viable fallback, but the journal states that neural networks should be central to submissions. Because the core theorem here is finite-memory construction geometry rather than specifically neural-network architecture, JMLR is the cleaner first fit.

The venue decision should be revisited only if journal policies materially change or JMLR declines the paper on scope/priority grounds.
