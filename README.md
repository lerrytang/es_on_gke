# Run Evolution Strategies on Google Kubernetes Engine

## Introduction

Evolution Strategies (ES) performs iterative optimization with a large population of trials that are usually distributedly conducted.
[Google Kubernetes Engine](https://cloud.google.com/kubernetes-engine/) (GKE) serves as a good platform for ES.  
We hope the instructions and code here serves as a quickstart for researchers to run their ES experiments on GKE.
Please refer to the blog [here] for more information about the repository.  
You are also strongly recommended to read this [blog](http://blog.otoro.net/2017/10/29/visual-evolution-strategies/) that provides excellent explanations if you want to know more about ES.


![Learning time comparison in BipedalWalkerHardcore](https://storage.googleapis.com/gcp_blog/img/bipedal_time_comparison.png)
![Learning time comparison in MinitaurLocomotion](https://storage.googleapis.com/gcp_blog/img/minitaur_time_comparison.png)

## How to use the code

The ES algorithms we provide as samples are Parameter-exploring Policy Gradients (PEPG) and Covariance Matrix Adaptation (CMA).
You can play with them in Google Brain's [Minitaur Locomotion](https://github.com/bulletphysics/bullet3/tree/master/examples/pybullet/gym/pybullet_envs/minitaur/envs) and OpenAI's [BipedalWalkerHardcore-v2](https://github.com/openai/gym/wiki/Leaderboard#bipedalwalkerhardcore-v2).
You can also easily extend the code here to add your ES algorithms or change the configs to try the algorithms in your own environments.

### Run the demos on GKE

#### 1. Before you begin

You need a cluster on Google Cloud Platform (GCP) to run the demos, follow the instructions [here](https://cloud.google.com/kubernetes-engine/docs/how-to/creating-a-cluster) to create yours.  
We use the following commands / configs to create our cluster, feel free to change these to suit your needs.  
```Bash
GCLOUD_PROJECT={your-project-id}

gcloud container clusters create es-cluster \
--zone=us-central1-a \
--machine-type=n1-standard-64 \
--max-nodes=20 \
--min-nodes=17 \
--num-nodes=17 \
--enable-autoscaling \
--project ${GCLOUD_PROJECT}
```

#### 2. Follow the instructions to deploy a demo

The following command builds a container image for you.
You need to generate a new image if you have changed the code, remember to change the image version number when you do so.  
```Bash
gcloud builds submit \
--tag gcr.io/${GCLOUD_PROJECT}/es-on-gke:1.0 . \
--timeout 3600 \
--project ${GCLOUD_PROJECT}
```

When the container image is built, edit `yaml/deploy_workers.yaml` and `yaml/deploy_master.yaml` to
* replace the `spec.template.spec.containers.image` with the one you just built
* change the `--config` parameter in `spec.template.spec.containers.command` to the environment you want to run

Replace `${GCLOUD_PROJECT}` in the following 2 yaml files with your project ID,
then start the ES workers and the ES master:
```Bash
# Run these commands to start workers.
sed "s/\${GCLOUD_PROJECT}/${GCLOUD_PROJECT}/g" yaml/deploy_workers_bipedal.yaml > workers.yaml
kubectl apply -f workers.yaml

# When all the workers are running, run these command to start the master.
sed "s/\${GCLOUD_PROJECT}/${GCLOUD_PROJECT}/g" yaml/deploy_master_bipedal.yaml > master.yaml
kubectl apply -f master.yaml
```
After a while you should be able to see your pods started in GCP console:  
![Pod started](https://storage.googleapis.com/gcp_blog/img/start_master_workers.png) 

That's all! ES should be training in your specified environment on GKE now.

#### 3. Check training progress and results

We provide 3 ways for you to check the training progress:
1. **Stackdriver** In GCP console, clicking GKE's Workloads page gives your detailed status report of your pods.
Go to the details of the `es-master-pod` and you can find "Container logs" there that will direct you to the Stackdriver's logging where you can see training and test rewards.
2. **HTTP Server** In our code, we start a simple HTTP server in the master to make training logs easily accessible to you.
You can access by checking the endpoint in `es-master-service` located in GKE's Services page. (The server may need some time to start up.)
3. **Kubectl** Finally you can use the *kubectl* command to fetch logs and models.
The following commands serve as examples.

```bash
POD_NAME=$(kubectl get pod | grep es-master | awk '{print $1}')

# Download reward vs time plot.
kubectl cp $POD_NAME:/var/log/es/log/reward_vs_time.png $HOME/
# Download reward vs iteration plot.
kubectl cp $POD_NAME:/var/log/es/log/reward_vs_iteration.png $HOME/
# Download best model so far.
kubectl cp $POD_NAME:/var/log/es/log/best_model.npz $HOME/
# Download model at iteration 1000.
kubectl cp $POD_NAME:/var/log/es/log/model_1000.npz $HOME/
# Download all test scores.
kubectl cp $POD_NAME:/var/log/es/log/scores.csv $HOME/
# Download everything (including models from all evaluation iterations).
kubectl cp $POD_NAME:/var/log/es/log $HOME/es_log
```


### Run the demos locally

As a debugging process, both training and test can be run locally.  
Use `train_local.sh` and `test.py`, and add proper options to do so.
```Bash
# train locally
bash ./train_local.sh -c {path-to-config-file} -n {number-of-workers}

# test locally
python test.py --logdir={path-to-log-directory}
```

### Clean up

When the tasks are done, you can download all the training logs and models for future analysis.  
Finally, if you don't need to run any tasks, don't forget to [delete the cluster](https://cloud.google.com/dataproc/docs/guides/manage-cluster).
