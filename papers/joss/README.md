# JOSS Paper Notes

This subdirectory contains the source files for a paper we plan to submit to the Journal of Open Source Software (JOSS).

If you want to locally edit and compile this paper (to preview the PDF that reviewers will generate from the source files), the
easiest way is to run the following docker command from the root directory of this repository.

```bash
docker run --rm --volume $PWD/paper:/data --user $(id -u):$(id -g) --env JOURNAL=joss openjournals/inara
```

If you do not have docker installed yet, you might have to run some command like the following to get docker installed
and runnable as regular non-root user (not guaranteed to work for all environments, but works on `fresh01.01101.dev` development server).

```bash
sudo snap connect docker:home
sudo addgroup --system docker
sudo adduser $USER docker
newgrp docker
sudo snap disable docker
sudo snap enable docker
```