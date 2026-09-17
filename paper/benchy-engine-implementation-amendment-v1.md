# Benchy Engine 1.0 — Implementation Amendment

Apply these changes to the earlier v10 / handoff v1 / spec v1 documents.

1. **Artifact paths are an Engine 1.0 representation, not a semantic invariant.**
   Hosted storage may be local disk, S3, GCS, MinIO, etc. Before core execution, assets are materialized into an allowed benchmark workspace. Dataset artifact references resolve to local absolute paths inside that workspace.

2. **Define path bases.**
   - `data.path`, prompt paths, and other benchmark-owned paths resolve from the benchmark workspace root.
   - Artifact paths inside JSONL resolve from the JSONL directory.
   - Canonicalized paths must remain inside the allowed workspace; reject traversal and symlink escape.

3. **Every AI-system goes through an adapter.**
   `type: model` is not a special engine execution branch. Reusable provider adapters implement the same `async invoke(input_object) -> output_object` protocol as all other integrations.

4. **Ontology lookup is runtime infrastructure.**
   YAML pins `ontology_version`; it does not provide an ontology filepath. Runtime resolves the matching installed/configured registry version.

5. **Concurrency must be observationally deterministic.**
   Work may execute concurrently, but results retain dataset indices and are serialized in dataset order.

6. **Timeouts/credentials/retries/concurrency/caching are runtime policy.**
   They do not belong in benchmark semantics. A timeout during an example invocation is an `execution_error`.

These amendments do not change Benchy's benchmark semantics, scoring model, compiler invariant, JSON IR role, or adapter boundary.
