## SyncReplicasOptimizer

The following accumulators/queue are created: N gradient accumulators, one per variable to train. Gradients are pushed to them and the chief worker will wait until enough gradients are collected and then average them before applying to variables. The accumulator will drop all stale gradients (more details in the accumulator op). 1 token queue where the optimizer pushes the new global_step value after all variables are updated.

The following local variable is created: * sync_rep_local_step, one per replica. Compared against the global_step in each accumulator to check for staleness of the gradients.


The optimizer adds nodes to the graph to collect gradients and pause the trainers until variables are updated. 

**For the Parameter Server job:** 1. An accumulator is created for each variable, and each replica pushes the gradients into the accumulators instead of directly applying them to the variables. 2. Each accumulator averages once enough gradients (replicas_to_aggregate) have been accumulated. 3. Apply the averaged gradients to the variables. 4. Only after all variables have been updated, increment the global step. 5. Only after step 4, pushes global_step in the token_queue, once for each worker replica. The workers can now fetch the global step, use it to update its local_step variable and start the next batch.

**For the replicas:** 1. Start a step: fetch variables and compute gradients. 2. Once the gradients have been computed, push them into gradient accumulators. Each accumulator will check the staleness and drop the stale. 3. After pushing all the gradients, dequeue an updated value of global_step from the token queue and record that step to its local_step variable. Note that this is effectively a barrier. 4. Start the next batch.


## tf.train.replica_device_setter

replica_device_setter 自动将Variable分配到ps tasks，round-robin分配策略


```python
# To build a cluster with two ps jobs on hosts ps0 and ps1, and 3 worker
# jobs on hosts worker0, worker1 and worker2.
cluster_spec = {
    "ps": ["ps0:2222", "ps1:2222"],
    "worker": ["worker0:2222", "worker1:2222", "worker2:2222"]}
with tf.device(tf.train.replica_device_setter(cluster=cluster_spec)):
  # Build your graph
  v1 = tf.Variable(...)  # assigned to /job:ps/task:0
  v2 = tf.Variable(...)  # assigned to /job:ps/task:1
  v3 = tf.Variable(...)  # assigned to /job:ps/task:0
# Run compute
```

常用用法：

命令行启动时加上环境变量：CUDA_VISIBLE_DEVICES=n，tensorflow中会看到/gpu:0设备 （https://github.com/tensorflow/tensorflow/issues/3234#issuecomment-236611040）

```python
tf.train.replica_device_setter(
    cluster=cluster_spec,
    worker_device='/job:worker/task:%d/gpu:0' % FLAGS.task_id
)
```

## tf.cond

```python
z = tf.multiply(a, b)
result = tf.cond(x < y, lambda: tf.add(x, z), lambda: tf.square(y))
```

z is needed for at least one branch of the cond, <font color=red>the tf.mul operation is always executed unconditionally</font>. Although this behavior is consistent with the dataflow model of TensorFlow, it has occasionally surprised some users who expected a lazier semantics. (https://www.tensorflow.org/versions/r1.0/api_docs/python/tf/cond)

再来看一个stackoverflow上的例子(https://stackoverflow.com/questions/37063952/confused-by-the-behavior-of-tf-cond)：

```python
pred = tf.constant(True)
x = tf.Variable([1])
assign_x_2 = tf.assign(x, [2])
def update_x_2():
  with tf.control_dependencies([assign_x_2]):
    return tf.identity(x)
y = tf.cond(pred, update_x_2, lambda: tf.identity(x))
with tf.Session() as session:
  session.run(tf.initialize_all_variables())
  print(y.eval())
```

不论pred设为True还是False，都得到y=[2]

稍微修改一下：

```python
pred = tf.placeholder(tf.bool, shape=[])
x = tf.Variable([1])
def update_x_2():
  with tf.control_dependencies([tf.assign(x, [2])]):
    return tf.identity(x)
y = tf.cond(pred, update_x_2, lambda: tf.identity(x))
with tf.Session() as session:
  session.run(tf.initialize_all_variables())
  print(y.eval(feed_dict={pred: False}))  # ==> [1]
  print(y.eval(feed_dict={pred: True}))   # ==> [2]
```

注意`tf.assign(x, [2])`被放到函数`update_x_2`内部，就能得到正确结果了

### 解释

tf.cond的两个函数fn_true和fn_false，在执行时创建的graph，看成tf.cond的两个内部graph，tf.cond会根据pred选择执行其中一个；但如果依赖tf.cond之前的op（看成tf.cond的外部graph），则该不论pred选择哪一个分支，都会执行此op。

所以，<font color=red>关键看tf.cond依赖的op在外部还是内部</font>，如果在外部，不论pred如何选择，都会执行，如果在内部，则根据pred只执行其中一个





