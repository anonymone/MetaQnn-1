#!/bin/bash

# examples: 
#	1) ./convert_script.sh cifar10

IR_DIR="./IR"
CONVERTED_MODEL_DIR="./Keras"
MODEL_NAME=$1
CODE="code_"$1

mkdir -p $IR_DIR
mkdir -p $CODE
mkdir -p $CONVERTED_MODEL_DIR

rm -f $MODEL_NAME.architectures
find ../$MODEL_NAME/trained_models -name \*train_net.prototxt sort >> $MODEL_NAME.architectures
readarray -t models < $MODEL_NAME.architectures

for i in "${!architectures[@]}"; do 
	mmtoir -f caffe -n ${architectures[$i]} --inputShape 3,32,32 -d $IR_DIR/$MODEL_NAME$i
	mmtocode -f keras -n $IR_DIR/$MODEL_NAME$i.pb -w $IR_DIR/$MODEL_NAME$i.npy -d $CODE/$MODEL_NAME$i.py
	# fix code
	sed -i 's/        weights_dict = np.load(weight_file).item()/        weights_dict = np.load(weight_file, allow_pickle = True).item()/' $CODE/$MODEL_NAME$i.py
	sed -i 's/    set_layer_weights(model, weights_dict)/    #set_layer_weights(model, weights_dict)/' $CODE/$MODEL_NAME_$i.py
	sed -i "s/weights_dict = np.load(weight_file, encoding='bytes').item()/weights_dict = None/" $CODE/$MODEL_NAME_$i.py
	mmtomodel -f keras -in$CODE/$MODEL_NAME_$i.py -iw $CODE/$MODEL_NAME_$i.py -o $CONVERTED_MODEL_DIR/$CODE/$MODEL_NAME_$i.h5 --dump_tag SERVING
done
