# JIT-Fine research data

Raw research data is deliberately not committed or copied into the production image.

Download `data.zip` from the official [JIT-Fine replication repository](https://github.com/jacknichao/JIT-Fine), verify SHA-256
`9e5ca1a393b70ee7e87c410b162005958775f3f3732f9f83da9dd24a7dfe2b47`, and extract
`data/jitline` outside this repository. The trainer verifies the seven required pickle-file
checksums recorded in `ml_training/src/constants.py`.

Pickle can execute code while loading. Run the trainer only in a disposable, read-only,
no-network container. Never load the supplied files on a production or shared host.

Licence review on 2026-08-30 found no licence file or GitHub licence declaration in the
JIT-Fine repository. This repository therefore does not redistribute the raw data. Obtain
legal/owner approval before redistributing source data or treating repository visibility as
permission.
