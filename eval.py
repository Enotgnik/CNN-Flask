#! /usr/bin/env python3

import tensorflow as tf
import numpy as np
import os
import sys
import data_processor
from tensorflow.contrib import learn


# Parameters
# ==================================================
#During retraining, this list was printed out, please replace this with that print out
_CLASSES = ['Exchange Email', 'Software', 'Universal Email', 'New Hire', 'LAN Account', 'Hardware - Other', 'Infrastructure', 'Employee Separation', 'iAPARS', 'Facilities', 'Laptop', 'Desktop', 'SharePoint', 'Telecom', 'Help Desk', 'Intranet Applications', 'AMI', 'Procurement', 'Intranet Account', 'Tracer', 'Self Service Issues', 'Change Management', 'Printing', 'Accounts Payable Inquiry', 'Catalyst', 'Track-IT', 'CAI-U University', 'Corpsys', 'inquiry', 'Tablet', 'Capriza', 'External Support', 'ITMPI', 'Associate Survey', 'Threat Events', 'CAI Website Inquiry', 'MI']

#Interactive session
tf.flags.DEFINE_boolean("interactive", False, "Init interactive session")

# Data Parameters
tf.flags.DEFINE_string("data_file", "test.csv", "Data source for the data.")

# Eval Parameters
tf.flags.DEFINE_integer("batch_size", 64, "Batch Size (default: 64)")
tf.flags.DEFINE_string("checkpoint_dir", "", "Checkpoint directory from training run")
tf.flags.DEFINE_boolean("eval_train", False, "Evaluate on all training data")

# Misc Parameters
tf.flags.DEFINE_boolean("allow_soft_placement", True, "Allow device soft device placement")
tf.flags.DEFINE_boolean("log_device_placement", False, "Log placement of ops on devices")


FLAGS = tf.flags.FLAGS
FLAGS._parse_flags()
print("\nParameters:")
for attr, value in sorted(FLAGS.__flags.items()):
    print("{}={}".format(attr.upper(), value))
print("")

# CHANGE THIS: Load data. Load your own data here
if FLAGS.eval_train:
    x_raw, y_test = data_processor.load_data_and_labels(FLAGS.data_file,True)
else:
    x_raw = ["User unable to complete self evaluation", "May I request the lists of active AD accounts and inactive/disabled AD accounts across all domains"]
    y_test = ['iAPARS', 'LAN Account']


# Evaluation
# ==================================================
def get_model_api():
    def model(txt):
        # Map data into vocabulary
        checkpoint = './runs/1518619812/checkpoints/'
        vocab_path = os.path.join(checkpoint, "..", "vocab")
        vocab_processor = learn.preprocessing.VocabularyProcessor.restore(vocab_path)
       # x_test = np.array(list(vocab_processor.transform(x_raw)))

        print("\nEvaluating...\n")

        checkpoint_file = tf.train.latest_checkpoint(checkpoint)
        graph = tf.Graph()

        with graph.as_default():
            session_conf = tf.ConfigProto(
              allow_soft_placement=FLAGS.allow_soft_placement,
              log_device_placement=FLAGS.log_device_placement)
            sess = tf.Session(config=session_conf)
            with sess.as_default():
                # Load the saved meta graph and restore variables
                saver = tf.train.import_meta_graph("{}.meta".format(checkpoint_file))
                saver.restore(sess, checkpoint_file)

                # Get the placeholders from the graph by name
                input_x = graph.get_operation_by_name("input_x").outputs[0]
                # input_y = graph.get_operation_by_name("input_y").outputs[0]
                dropout_keep_prob = graph.get_operation_by_name("dropout_keep_prob").outputs[0]

                # Tensors we want to evaluate
                predictions = graph.get_operation_by_name("output/predictions").outputs[0]

                x_raw = []
                #y_test=None
                x_raw.append(txt)
                x_test = np.array(list(vocab_processor.transform(x_raw)))

                batches = data_processor.batch_iter(list(x_test), FLAGS.batch_size, 1, shuffle=False)

                # Collect the predictions here
                all_predictions = []

                for x_test_batch in batches:
                    batch_predictions = sess.run(predictions, {input_x: x_test_batch, dropout_keep_prob: 1.0})
                    batch_predictions=[_CLASSES[x] for x in batch_predictions]
                    all_predictions = np.concatenate([all_predictions, batch_predictions])
                    print(batch_predictions)
                    for i in range(len(x_raw)):
                        output_data = {"input": txt , "output": all_predictions[i]}
                        print(output_data)
                        return output_data
                        #snService.createINC(txt, all_predictions[i])
    return model

