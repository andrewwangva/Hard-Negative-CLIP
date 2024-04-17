#### Local setup using Docker

You need to take the following steps to get `al-folio` up and running in your local machine:

- First, [install docker](https://docs.docker.com/get-docker/) (Install Docker Desktop). Make sure that Docker Destop is open/running on your computer, and if it hangs you may need to restart your computer once after the installation
- Then, clone this repository to your machine:

```bash
$ git clone git@github.com:<your-username>/<your-repo-name>.git
$ cd <your-repo-name>
```

Finally, run the following command that will pull a pre-built image from DockerHub and will run your website.

```bash
$ ./bin/dockerhub_run.sh
```
