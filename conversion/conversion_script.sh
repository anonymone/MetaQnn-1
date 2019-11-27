#!/bin/bash

# You can either pass one model or all the models that will be then dataset's directory
# To use it with many just pass as argument after script "many"
# To use it with one, pass as arguments after script on one /path/to/train_net.prototxt /path/to/caffemodel
# examples: 
#	1) ./convert_script.sh many cifar10
#	2) ./convert_script.sh one cifar10 ./train_net.prototxt ./modelsave_iter_3510.caffemodel

IR_DIR="./IR"
CONVERTED_MODEL_DIR="./Keras"
MODEL_NAME=$2
CODE="code"

mkdir -p $IR_DIR
mkdir -p $CODE
mkdir -p $CONVERTED_MODEL_DIR

if [ "$1" = "one" ]; then
	mmtoir -f caffe -n $3 -w $4 --inputShape 3,32,32 -d $IR_DIR/$MODEL_NAME
	mmtocode -f keras -n "$IR_DIR/$MODEL_NAME.pb" -w $IR_DIR/$MODEL_NAME.npy -d $CODE/$MODEL_NAME.py
	# fix code
	sed -i 's/        weights_dict = np.load(weight_file).item()/        weights_dict = np.load(weight_file, allow_pickle = True).item()/' $CODE/$MODEL_NAME.py
	sed -i 's/    set_layer_weights(model, weights_dict)/    #set_layer_weights(model, weights_dict)/' $CODE/$MODEL_NAME.py
	mmtomodel -f keras -in $CODE/$MODEL_NAME.py -iw $IR_DIR/$MODEL_NAME.npy -o $CONVERTED_MODEL_DIR/$MODEL_NAME.ckpt --dump_tag SERVING
else
	rm -f $MODEL_NAME.architectures
	rm -f $MODEL_NAME.models
	find ../$MODEL_NAME/trained_models -name \*train_net.prototxt >> $MODEL_NAME.architectures
	find ../$MODEL_NAME/trained_models -name \*.caffemodel >> $MODEL_NAME.models
	python ./get_latest_models.py $MODEL_NAME.architectures $MODEL_NAME.models
	readarray -t architectures < $MODEL_NAME.architectures
	readarray -t models < $MODEL_NAME.models

	for i in "${!models[@]}"; do 
		mmtoir -f caffe -n ${architectures[$i]} -w ${models[$i]} --inputShape 3,32,32 -d $IR_DIR/$MODEL_NAME$i
		mmtocode -f keras -n $IR_DIR/$MODEL_NAME$i.pb -w $IR_DIR/$MODEL_NAME$i.npy -d $CODE/$MODEL_NAME$i.py
		# fix code
		sed -i 's/        weights_dict = np.load(weight_file).item()/        weights_dict = np.load(weight_file, allow_pickle = True).item()/' $CODE/$MODEL_NAME$i.py
		sed -i 's/    set_layer_weights(model, weights_dict)/    #set_layer_weights(model, weights_dict)/' $CODE/$MODEL_NAME$i.py
		mmtomodel -f keras -in $CODE/$MODEL_NAME$i.py -iw $IR_DIR/$MODEL_NAME$i.npy -o $CONVERTED_MODEL_DIR/$MODEL_NAME$i.ckpt --dump_tag SERVING
	done
fi
