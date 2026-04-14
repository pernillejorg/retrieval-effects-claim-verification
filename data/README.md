# Data

This folder is for dataset loading and preprocessing. Raw data files are not committed to the repo — they get downloaded locally using the scripts in each subfolder.

---

## Dataset 1: SciFact (Primary)

Source: [allenai/scifact](https://huggingface.co/datasets/allenai/scifact) on Hugging Face
Paper: Wadden et al. (2020), *Fact or Fiction: Verifying Scientific Claims*
Task: classify a scientific claim as SUPPORT, CONTRADICT, or NOT ENOUGH INFO using a corpus of paper abstracts as evidence

### Why SciFact?
SciFact is the main dataset for this project because retrieval is genuinely difficult here. The evidence corpus has ~5,000 abstracts and the claims are precise enough that topically similar but non-committal documents are a real problem, which is exactly the motivation for the stance reranker. My supervisor specifically recommended it for this reason.

### Structure
- train: 809 claims with labelled evidence
- - validation: 300 claims
  - - corpus: 5,183 paper abstracts (the retrieval corpus)
   
    - ### Labels
   
    - | Label | Meaning |
    - |---|---|
    - | SUPPORT | the evidence supports the claim |
    - | CONTRADICT | the evidence contradicts the claim |
    - | NOT_ENOUGH_INFO | no evidence found that directly addresses the claim |
   
    - ### How to load it
    - ```python
      from datasets import load_dataset
      dataset = load_dataset("allenai/scifact")
      corpus = load_dataset("allenai/scifact", "corpus")
      ```

      ---

      ## Dataset 2: SciClaimHunt (Secondary)

      Source: to be confirmed — check Hugging Face and recent ACL/EMNLP proceedings
      Task: scientific claim verification, similar label structure to SciFact

      ### Why SciClaimHunt?
      The whole point of using a second dataset is to check whether the findings from SciFact actually generalise or whether they are specific to that corpus. Running the core pipeline variants on both means I can make claims like "stance reranking helps consistently" rather than just "stance reranking helped on SciFact." My supervisor asked for this specifically. If the results diverge between datasets that is also an interesting and valid finding.

      The plan is to do full manual failure annotation on SciFact only, and use SciClaimHunt for quantitative comparison across conditions.

      ### Key differences from SciFact
      - more recent dataset with a different source corpus
      - - allows cross-dataset comparison of failure patterns and stance reranking effectiveness
       
        - ---

        ## A note on raw data files

        Raw data (JSON, JSONL, etc.) is excluded from version control via `.gitignore`. This keeps the repo lightweight and avoids any issues with redistributing dataset files. Download and cache them locally using the loading scripts.

        ---

        ## Folder structure once data is downloaded locally

        ```
        data/
        ├── README.md               # this file
        ├── scifact/
        │   ├── download.py         # script to download and cache SciFact
        │   └── (cached files)      # gitignored
        └── sciclaimhunt/
            ├── download.py         # script to download and cache SciClaimHunt
            └── (cached files)      # gitignored
        ```
