#include <stdlib.h>
#include <superneurons.h>
using namespace SuperNeurons;

int main(int argc, char **argv) {
    char* train_label_bin;
    char* train_image_bin;
    char* test_label_bin;
    char* test_image_bin;
    char* train_mean_file;
    network_t<float> n(0.001, 0.0005, 0.9, 3000);


    train_mean_file = (char *) "/home/xluo/DeepLearning/data/cifar10/cifar10_train_image.mean";
    train_image_bin = (char *) "/home/xluo/DeepLearning/data/cifar10/cifar10_train_image.bin";
    train_label_bin = (char *) "/home/xluo/DeepLearning/data/cifar10/cifar10_train_label.bin";
    test_image_bin  = (char *) "/home/xluo/DeepLearning/data/cifar10/cifar10_test_image.bin";
    test_label_bin  = (char *) "/home/xluo/DeepLearning/data/cifar10/cifar10_test_label.bin";
    const size_t batch_size = 100; //train and test must be same
    parallel_reader_t<float > reader2(test_image_bin, test_label_bin, 2, batch_size, train_mean_file);
    base_layer_t<float>* data_2 = (base_layer_t<float>*) new data_layer_t<float>(data_test, &reader2);
    parallel_reader_t<float > reader1(train_image_bin, train_label_bin, 2, batch_size, train_mean_file);
    base_layer_t<float>* data_1 = (base_layer_t<float>*) new data_layer_t<float>(data_train, &reader1);

    /*
     conv_layer_t(size_t num_output_,
     size_t kernel_size_,
     size_t stride_,
     size_t padding_h_,
     size_t padding_w_,
     bool enable_bias);
     */
    //if the dims of H,W after conv is not reduced, pad with half the filter sizes (round down). 3/2 = 1.5 = 1;
    base_layer_t<float>* conv_1 = (base_layer_t<float>*) new conv_layer_t<float> (16, 3, 1, 0, 0, false);
    base_layer_t<float>* act_1  = (base_layer_t<float>*) new act_layer_t<float>(CUDNN_ACTIVATION_RELU,CUDNN_NOT_PROPAGATE_NAN);
    //residual unit 1
    base_layer_t<float>* fork_1 = (base_layer_t<float>*) new fork_layer_t<float>();

    base_layer_t<float>* conv_2 = (base_layer_t<float>*) new conv_layer_t<float> (16, 3, 1, 1, 1, false);
    base_layer_t<float>* act_2  = (base_layer_t<float>*) new act_layer_t<float>(CUDNN_ACTIVATION_RELU,CUDNN_NOT_PROPAGATE_NAN);

    base_layer_t<float>* conv_3 = (base_layer_t<float>*) new conv_layer_t<float> (16, 3, 1, 1, 1, false);



    base_layer_t<float>* join_1 = (base_layer_t<float>*) new join_layer_t<float>();
    base_layer_t<float>* ja_1   = (base_layer_t<float>*) new act_layer_t<float>(CUDNN_ACTIVATION_RELU,CUDNN_NOT_PROPAGATE_NAN);

    //residual unit 2
    base_layer_t<float>* fork_2  = (base_layer_t<float>*) new fork_layer_t<float>();

    base_layer_t<float>* conv_4  = (base_layer_t<float>*) new conv_layer_t<float> (16, 3, 1, 1, 1, false);
    base_layer_t<float>* act_3   = (base_layer_t<float>*) new act_layer_t<float>(CUDNN_ACTIVATION_RELU,CUDNN_NOT_PROPAGATE_NAN);


    base_layer_t<float>* conv_5  = (base_layer_t<float>*) new conv_layer_t<float> (16, 3, 1, 1, 1, false);



    base_layer_t<float>* join_2  = (base_layer_t<float>*) new join_layer_t<float>();
    base_layer_t<float>* ja_2    = (base_layer_t<float>*) new act_layer_t<float>(CUDNN_ACTIVATION_RELU,CUDNN_NOT_PROPAGATE_NAN);

    //residual unit 3
    base_layer_t<float>* fork_3  = (base_layer_t<float>*) new fork_layer_t<float>();

    base_layer_t<float>* conv_6  = (base_layer_t<float>*) new conv_layer_t<float> (16, 3, 1, 1, 1, false);
    base_layer_t<float>* act_4   = (base_layer_t<float>*) new act_layer_t<float>(CUDNN_ACTIVATION_RELU,CUDNN_NOT_PROPAGATE_NAN);

    base_layer_t<float>* conv_7  = (base_layer_t<float>*) new conv_layer_t<float> (16, 3, 1, 1, 1, false);



    base_layer_t<float>* join_3  = (base_layer_t<float>*) new join_layer_t<float>();
    base_layer_t<float>* ja_3    = (base_layer_t<float>*) new act_layer_t<float>(CUDNN_ACTIVATION_RELU,CUDNN_NOT_PROPAGATE_NAN);

    //residual unit 4
    base_layer_t<float>* fork_4  = (base_layer_t<float>*) new fork_layer_t<float>();

    base_layer_t<float>* conv_8  = (base_layer_t<float>*) new conv_layer_t<float> (16, 3, 1, 1, 1, false);

    base_layer_t<float>* act_5   = (base_layer_t<float>*) new act_layer_t<float>(CUDNN_ACTIVATION_RELU,CUDNN_NOT_PROPAGATE_NAN);

    base_layer_t<float>* conv_9  = (base_layer_t<float>*) new conv_layer_t<float> (16, 3, 1, 1, 1, false);



    base_layer_t<float>* join_4  = (base_layer_t<float>*) new join_layer_t<float>();
    base_layer_t<float>* ja_4    = (base_layer_t<float>*) new act_layer_t<float>(CUDNN_ACTIVATION_RELU,CUDNN_NOT_PROPAGATE_NAN);

    //residual unit 5
    base_layer_t<float>* fork_5  = (base_layer_t<float>*) new fork_layer_t<float>();

    base_layer_t<float>* conv_10  = (base_layer_t<float>*) new conv_layer_t<float> (16, 3, 1, 1, 1, false);

    base_layer_t<float>* act_6   = (base_layer_t<float>*) new act_layer_t<float>(CUDNN_ACTIVATION_RELU,CUDNN_NOT_PROPAGATE_NAN);

    base_layer_t<float>* conv_11  = (base_layer_t<float>*) new conv_layer_t<float> (16, 3, 1, 1, 1, false);



    base_layer_t<float>* join_5  = (base_layer_t<float>*) new join_layer_t<float>();
    base_layer_t<float>* ja_5    = (base_layer_t<float>*) new act_layer_t<float>(CUDNN_ACTIVATION_RELU,CUDNN_NOT_PROPAGATE_NAN);

    //residual unit 6
    base_layer_t<float>* fork_6  = (base_layer_t<float>*) new fork_layer_t<float>();
    //right
    base_layer_t<float>* conv_12  = (base_layer_t<float>*) new conv_layer_t<float> (32, 3, 2, 1, 1, false);

    base_layer_t<float>* act_7   = (base_layer_t<float>*) new act_layer_t<float>(CUDNN_ACTIVATION_RELU,CUDNN_NOT_PROPAGATE_NAN);

    base_layer_t<float>* conv_13  = (base_layer_t<float>*) new conv_layer_t<float> (32, 3, 1, 1, 1, false);


    //left
    base_layer_t<float>* conv_14  = (base_layer_t<float>*) new conv_layer_t<float> (32, 1, 2, 0, 0, false);


    base_layer_t<float>* join_6  = (base_layer_t<float>*) new join_layer_t<float>();
    base_layer_t<float>* ja_6    = (base_layer_t<float>*) new act_layer_t<float>(CUDNN_ACTIVATION_RELU,CUDNN_NOT_PROPAGATE_NAN);

    //residual unit 7
    base_layer_t<float>* fork_7  = (base_layer_t<float>*) new fork_layer_t<float>();

    base_layer_t<float>* conv_15  = (base_layer_t<float>*) new conv_layer_t<float> (32, 3, 1, 1, 1, false);

    base_layer_t<float>* act_8   = (base_layer_t<float>*) new act_layer_t<float>(CUDNN_ACTIVATION_RELU,CUDNN_NOT_PROPAGATE_NAN);

    base_layer_t<float>* conv_16  = (base_layer_t<float>*) new conv_layer_t<float> (32, 3, 1, 1, 1, false);



    base_layer_t<float>* join_7  = (base_layer_t<float>*) new join_layer_t<float>();
    base_layer_t<float>* ja_7    = (base_layer_t<float>*) new act_layer_t<float>(CUDNN_ACTIVATION_RELU,CUDNN_NOT_PROPAGATE_NAN);

    //residual unit 8
    base_layer_t<float>* fork_8  = (base_layer_t<float>*) new fork_layer_t<float>();

    base_layer_t<float>* conv_17  = (base_layer_t<float>*) new conv_layer_t<float> (32, 3, 1, 1, 1, false);

    base_layer_t<float>* act_9   = (base_layer_t<float>*) new act_layer_t<float>(CUDNN_ACTIVATION_RELU,CUDNN_NOT_PROPAGATE_NAN);

    base_layer_t<float>* conv_18  = (base_layer_t<float>*) new conv_layer_t<float> (32, 3, 1, 1, 1, false);



    base_layer_t<float>* join_8  = (base_layer_t<float>*) new join_layer_t<float>();
    base_layer_t<float>* ja_8    = (base_layer_t<float>*) new act_layer_t<float>(CUDNN_ACTIVATION_RELU,CUDNN_NOT_PROPAGATE_NAN);

    //residual unit 9
    base_layer_t<float>* fork_9  = (base_layer_t<float>*) new fork_layer_t<float>();

    base_layer_t<float>* conv_19  = (base_layer_t<float>*) new conv_layer_t<float> (32, 3, 1, 1, 1, false);

    base_layer_t<float>* act_10   = (base_layer_t<float>*) new act_layer_t<float>(CUDNN_ACTIVATION_RELU,CUDNN_NOT_PROPAGATE_NAN);

    base_layer_t<float>* conv_20  = (base_layer_t<float>*) new conv_layer_t<float> (32, 3, 1, 1, 1, false);



    base_layer_t<float>* join_9  = (base_layer_t<float>*) new join_layer_t<float>();
    base_layer_t<float>* ja_9    = (base_layer_t<float>*) new act_layer_t<float>(CUDNN_ACTIVATION_RELU,CUDNN_NOT_PROPAGATE_NAN);

    //residual unit 10
    base_layer_t<float>* fork_10  = (base_layer_t<float>*) new fork_layer_t<float>();

    base_layer_t<float>* conv_21  = (base_layer_t<float>*) new conv_layer_t<float> (32, 3, 1, 1, 1, false);

    base_layer_t<float>* act_11   = (base_layer_t<float>*) new act_layer_t<float>(CUDNN_ACTIVATION_RELU,CUDNN_NOT_PROPAGATE_NAN);

    base_layer_t<float>* conv_22  = (base_layer_t<float>*) new conv_layer_t<float> (32, 3, 1, 1, 1, false);



    base_layer_t<float>* join_10  = (base_layer_t<float>*) new join_layer_t<float>();
    base_layer_t<float>* ja_10    = (base_layer_t<float>*) new act_layer_t<float>(CUDNN_ACTIVATION_RELU,CUDNN_NOT_PROPAGATE_NAN);

    //residual unit 11
    base_layer_t<float>* fork_11  = (base_layer_t<float>*) new fork_layer_t<float>();
    //right
    base_layer_t<float>* conv_23  = (base_layer_t<float>*) new conv_layer_t<float> (64, 3, 2, 1, 1, false);

    base_layer_t<float>* act_12   = (base_layer_t<float>*) new act_layer_t<float>(CUDNN_ACTIVATION_RELU,CUDNN_NOT_PROPAGATE_NAN);

    base_layer_t<float>* conv_24  = (base_layer_t<float>*) new conv_layer_t<float> (64, 3, 1, 1, 1, false);


    //left
    base_layer_t<float>* conv_25  = (base_layer_t<float>*) new conv_layer_t<float> (64, 1, 2, 0, 0, false);


    base_layer_t<float>* join_11  = (base_layer_t<float>*) new join_layer_t<float>();
    base_layer_t<float>* ja_11    = (base_layer_t<float>*) new act_layer_t<float>(CUDNN_ACTIVATION_RELU,CUDNN_NOT_PROPAGATE_NAN);

    //residual unit 12
    base_layer_t<float>* fork_12  = (base_layer_t<float>*) new fork_layer_t<float>();

    base_layer_t<float>* conv_26  = (base_layer_t<float>*) new conv_layer_t<float> (64, 3, 1, 1, 1, false);

    base_layer_t<float>* act_13   = (base_layer_t<float>*) new act_layer_t<float>(CUDNN_ACTIVATION_RELU,CUDNN_NOT_PROPAGATE_NAN);

    base_layer_t<float>* conv_27  = (base_layer_t<float>*) new conv_layer_t<float> (64, 3, 1, 1, 1, false);



    base_layer_t<float>* join_12  = (base_layer_t<float>*) new join_layer_t<float>();
    base_layer_t<float>* ja_12    = (base_layer_t<float>*) new act_layer_t<float>(CUDNN_ACTIVATION_RELU,CUDNN_NOT_PROPAGATE_NAN);

    //residual unit 13
    base_layer_t<float>* fork_13  = (base_layer_t<float>*) new fork_layer_t<float>();

    base_layer_t<float>* conv_28  = (base_layer_t<float>*) new conv_layer_t<float> (64, 3, 1, 1, 1, false);

    base_layer_t<float>* act_14   = (base_layer_t<float>*) new act_layer_t<float>(CUDNN_ACTIVATION_RELU,CUDNN_NOT_PROPAGATE_NAN);

    base_layer_t<float>* conv_29  = (base_layer_t<float>*) new conv_layer_t<float> (64, 3, 1, 1, 1, false);



    base_layer_t<float>* join_13  = (base_layer_t<float>*) new join_layer_t<float>();
    base_layer_t<float>* ja_13    = (base_layer_t<float>*) new act_layer_t<float>(CUDNN_ACTIVATION_RELU,CUDNN_NOT_PROPAGATE_NAN);

    //residual unit 14
    base_layer_t<float>* fork_14  = (base_layer_t<float>*) new fork_layer_t<float>();

    base_layer_t<float>* conv_30  = (base_layer_t<float>*) new conv_layer_t<float> (64, 3, 1, 1, 1, false);

    base_layer_t<float>* act_15   = (base_layer_t<float>*) new act_layer_t<float>(CUDNN_ACTIVATION_RELU,CUDNN_NOT_PROPAGATE_NAN);

    base_layer_t<float>* conv_31  = (base_layer_t<float>*) new conv_layer_t<float> (64, 3, 1, 1, 1, false);



    base_layer_t<float>* join_14  = (base_layer_t<float>*) new join_layer_t<float>();
    base_layer_t<float>* ja_14    = (base_layer_t<float>*) new act_layer_t<float>(CUDNN_ACTIVATION_RELU,CUDNN_NOT_PROPAGATE_NAN);

    //residual unit 15
    base_layer_t<float>* fork_15  = (base_layer_t<float>*) new fork_layer_t<float>();

    base_layer_t<float>* conv_32  = (base_layer_t<float>*) new conv_layer_t<float> (64, 3, 1, 1, 1, false);

    base_layer_t<float>* act_16   = (base_layer_t<float>*) new act_layer_t<float>(CUDNN_ACTIVATION_RELU,CUDNN_NOT_PROPAGATE_NAN);

    base_layer_t<float>* conv_33  = (base_layer_t<float>*) new conv_layer_t<float> (64, 3, 1, 1, 1, false);



    base_layer_t<float>* join_15  = (base_layer_t<float>*) new join_layer_t<float>();
    base_layer_t<float>* ja_15    = (base_layer_t<float>*) new act_layer_t<float>(CUDNN_ACTIVATION_RELU,CUDNN_NOT_PROPAGATE_NAN);



    base_layer_t<float>* pool_1   = (base_layer_t<float>*) new pool_layer_t<float>(CUDNN_POOLING_AVERAGE_COUNT_EXCLUDE_PADDING, CUDNN_NOT_PROPAGATE_NAN, 1, 1, 2, 2);
    base_layer_t<float>* full_conn_1 = (base_layer_t<float>*) new fully_connected_layer_t<float> (10, true);
    base_layer_t<float>* softmax = (base_layer_t<float>*) new softmax_layer_t<float>(CUDNN_SOFTMAX_ACCURATE, CUDNN_SOFTMAX_MODE_INSTANCE);

    //setup test
    data_2->hook_to( conv_1 );
    //setup network
    data_1->hook( conv_1 );
    conv_1->hook( act_1 );
    act_1->hook( fork_1 );

    //residual 1
    fork_1->hook( join_1 );
    fork_1->hook( conv_2 );
    conv_2->hook( act_2 );
    act_2->hook( conv_3 );
    conv_3->hook( join_1 );
    join_1->hook( ja_1 );
    ja_1->hook( fork_2 );

    //residual 2
    fork_2->hook(join_2);
    fork_2->hook(conv_4);
    conv_4->hook(act_3);
    act_3->hook(conv_5);
    conv_5->hook(join_2);
    join_2->hook(ja_2);
    ja_2->hook( fork_3 );

    //residual 3
    fork_3->hook(join_3);
    fork_3->hook(conv_6);
    conv_6->hook(act_4);
    act_4->hook(conv_7);
    conv_7->hook(join_3);
    join_3->hook(ja_3);
    ja_3->hook( fork_4 );

    //residual 4
    fork_4->hook(join_4);
    fork_4->hook(conv_8);
    conv_8->hook(act_5);
    act_5->hook(conv_9);
    conv_9->hook(join_4);
    join_4->hook(ja_4);
    ja_4->hook( fork_5 );

    //residual 5
    fork_5->hook(join_5);
    fork_5->hook(conv_10);
    conv_10->hook(act_6);
    act_6->hook(conv_11);
    conv_11->hook(join_5);
    join_5->hook(ja_5);
    ja_5->hook( fork_6 );

    //residual 6
    //   left
    fork_6->hook(conv_14);
    conv_14->hook(join_6);
    //   right
    fork_6->hook(conv_12);
    conv_12->hook(act_7);
    act_7->hook(conv_13);
    conv_13->hook(join_6);
    join_6->hook(ja_6);
    ja_6->hook( fork_7 );

    //residual 7
    fork_7->hook(join_7);
    fork_7->hook(conv_15);
    conv_15->hook(act_8);
    act_8->hook(conv_16);
    conv_16->hook(join_7);
    join_7->hook(ja_7);
    ja_7->hook( fork_8 );

    //residual 8
    fork_8->hook(join_8);
    fork_8->hook(conv_17);
    conv_17->hook(act_9);
    act_9->hook(conv_18);
    conv_18->hook(join_8);
    join_8->hook(ja_8);
    ja_8->hook( fork_9 );

    //residual 9
    fork_9->hook(join_9);
    fork_9->hook(conv_19);
    conv_19->hook(act_10);
    act_10->hook(conv_20);
    conv_20->hook(join_9);
    join_9->hook(ja_9);
    ja_9->hook( fork_10 );

    //residual 10
    fork_10->hook(join_10);
    fork_10->hook(conv_21);
    conv_21->hook(act_11);
    act_11->hook(conv_22);
    conv_22->hook(join_10);
    join_10->hook(ja_10);
    ja_10->hook( fork_11 );

    //residual 11
    //   left
    fork_11->hook(conv_25);
    conv_25->hook(join_11);
    //   right
    fork_11->hook(conv_23);
    conv_23->hook(act_12);
    act_12->hook(conv_24);
    conv_24->hook(join_11);
    join_11->hook(ja_11);
    ja_11->hook( fork_12 );

    //residual 12
    fork_12->hook(join_12);
    fork_12->hook(conv_26);
    conv_26->hook(act_13);
    act_13->hook(conv_27);
    conv_27->hook(join_12);
    join_12->hook(ja_12);
    ja_12->hook( fork_13 );

    //residual 13
    fork_13->hook(join_13);
    fork_13->hook(conv_28);
    conv_28->hook(act_14);
    act_14->hook(conv_29);
    conv_29->hook(join_13);
    join_13->hook(ja_13);
    ja_13->hook( fork_14 );

    //residual 14
    fork_14->hook(join_14);
    fork_14->hook(conv_30);
    conv_30->hook(act_15);
    act_15->hook(conv_31);
    conv_31->hook(join_14);
    join_14->hook(ja_14);
    ja_14->hook( fork_15 );

    //residual 15
    fork_15->hook(join_15);
    fork_15->hook(conv_32);
    conv_32->hook(act_16);
    act_16->hook(conv_33);
    conv_33->hook(join_15);
    join_15->hook(ja_15);


    ja_15->hook(pool_1);
    pool_1->hook(full_conn_1);
    full_conn_1->hook( softmax );

    n.fsetup(data_1);
    n.bsetup(softmax);
    n.setup_test( data_2, 100 );

    n.train(500000);
}
