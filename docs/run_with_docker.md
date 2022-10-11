
# Running with Docker on Windows or Linux with NVIDIA GPU
On Windows 10 or 11 you can follow this guide to setting up Docker with WSL2 https://www.youtube.com/watch?v=PB7zM3JrgkI

Native Linux

```
cd stablediffusion-infinity/docker
./docker-run.sh
```

Windows 10,11 with WSL2 shell:
- open windows Command Prompt, type "bash"
- once in bash, type:
```
cd /mnt/c/PATH-TO-YOUR/stablediffusion-infinity/docker
./docker-run.sh
```

Open "http://localhost:8888" in your browser ( even though the log says http://0.0.0.0:8888 )