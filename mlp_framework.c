#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>
#include <string.h>
#define MAX_INPUTS 100
#define MAX_OUTPUTS 100
#define LEARNING_RATE 0.005
#define MAX_NEURONS 100
#define MAX_LETTERS 20
#define MAX_DATA 500

typedef struct{
    float weight[MAX_INPUTS];
    float bias; 
    float output; // OUTPUT DE LA NEURONA
    float output_prereulu; // OUTPUT ABANS DE RELU
    float delta; // ERROR PER A BACKPROPAGATION
    int num_weights; // WEIGHTS REBUTS PER NEURONA
} Neuron;

typedef struct{
    Neuron neurons[MAX_NEURONS];
    int neuron_count;
    int inputs_per_neuron;
    float inputs_cache[MAX_INPUTS];
    float outputs_cache[MAX_INPUTS];
    float deltas_cache[MAX_INPUTS];
    int activation_type;
} Layer;

typedef struct{
    Layer layers[10];
    int layer_count;
} Network;

float relu_derivative(float x){
    if (x > 0) return 1.0;
    else return 0.01;
}

float tanh_derivative(float x){
    return (1 - x*x);
}
float sigmoid(float x){
    return 1.0/ (1.0 + exp(-x));
}

float sigmoid_derivative(float x){
    return x * (1- x);
}

void initialize_neuron(Neuron* n, int n_inputs){
    n->num_weights = n_inputs;
    for(int i = 0; i < n_inputs; i++){
        n->weight[i] = ((float)rand() / (float)RAND_MAX) * 2.0 - 1.0; // Pesos inicialitzats aleatòriament entre -0.5 i 0.5
        n->weight[i] *= sqrt(2.0 / n_inputs);
    }
    n->bias = ((float)rand() / (float)RAND_MAX);
}

float relu(float x){
    if (x>0){
        return x;
    }
    else return x * 0.01;
}

float activate(Neuron* n, float input[], int activation_type){
    n->output_prereulu = n->bias;
    for(int i = 0; i < n->num_weights; i++){
        n->output_prereulu += n->weight[i] * input[i];
    }
    if(activation_type == 0){
        n->output = relu(n->output_prereulu);
    }
    else n->output = sigmoid(n->output_prereulu);
    
    return n->output;
}

void initialize_layer(Layer* l, int neuron_count, int n_inputs){
    l->neuron_count = neuron_count;
    l->inputs_per_neuron = n_inputs;
    for(int i = 0; i < neuron_count; i++){
        initialize_neuron(&l->neurons[i], n_inputs);
    }
}

void initialize_network(Network* net, int layer_count, int neuron_count, int input_size_data, int output_neurons){
    net->layer_count = layer_count;
    initialize_layer(&net->layers[0], neuron_count, input_size_data);
    net->layers[0].activation_type = 0; // FUNCIÓ RELU
    int i;
    int inputs_from_prev_layer;
    for (i=1; i < layer_count-1; i++){
        inputs_from_prev_layer = net->layers[i-1].neuron_count;
        initialize_layer(&net->layers[i], neuron_count, inputs_from_prev_layer);
         net->layers[i].activation_type = 0; //FUNCIÓ RELU A LA RESTA (ocultes)
        
    }
    inputs_from_prev_layer = net->layers[i-1].neuron_count;
    initialize_layer(&net->layers[i], output_neurons, inputs_from_prev_layer);
    net->layers[i].activation_type = 1;
}



void forward_layer(float inputs[MAX_INPUTS], Layer* layer){
    for(int i=0; i<layer->neuron_count;i++){
        layer->neurons[i].output = activate(&layer->neurons[i], inputs, layer->activation_type);
        layer->outputs_cache[i] = layer->neurons[i].output;
    }
    for(int i = 0; i < layer->inputs_per_neuron; i++ ){ 
        layer->inputs_cache[i] = inputs[i];
    }
}

void forward_Network(Network* net, float inputs[MAX_INPUTS]){
    forward_layer(inputs, &net->layers[0]);
    for(int i = 1; i < net->layer_count; i++){
        float* inputs_next = net->layers[i-1].outputs_cache;
        forward_layer(inputs_next, &net->layers[i]);
    }
}

void backpropagation(Network* net, float expected_outputs[MAX_OUTPUTS], int outputs_size){
    float error;
    Layer* output_layer = &net->layers[net->layer_count - 1]; // CAPA FINAL
    for(int i = 0; i < outputs_size; i++){ //PER A CADA OUTPUT, EN CAS DE QUE SIGUI UN VECTOR
            float real_output = output_layer->outputs_cache[i]; // MIREM EL OUTPUT QUE HA PREDIT LA IA
            float expected_output = expected_outputs[i]; // VALOR QUE HAURIA D'HAVER SORTIT
            error = expected_output - real_output; // CALCULEM DIFERENCIA
            float real_output_derivative = sigmoid_derivative   (real_output); // DERIVADA
            output_layer->deltas_cache[i] = error * real_output_derivative;
    }
    for(int i = net->layer_count -2; i >= 0; i--){
        Layer* actual_layer = &net->layers[i];
        Layer* next_layer = &net->layers[i+1];
        for(int j = 0; j < actual_layer->neuron_count; j++){
            float spread_error = 0.0;
                for(int k = 0; k<next_layer->neuron_count; k++){
                    float deltak = next_layer->deltas_cache[k];
                    float weightkj = next_layer->neurons[k].weight[j];
                    spread_error += deltak * weightkj;
                    
                }
                float derivative = relu_derivative(actual_layer->outputs_cache[j]);
                actual_layer->deltas_cache[j] = spread_error * derivative;
        }    
        
    }
}

void update_weights(Network* net){
    for(int i = 0; i< net->layer_count; i++){
        Layer* actual_layer = &net->layers[i]; 
        for(int j=0; j< actual_layer->neuron_count; j++){
            Neuron* actual_neuron = &actual_layer->neurons[j];
            float delta = actual_layer->deltas_cache[j];
            for(int k=0; k< actual_neuron->num_weights; k++){
                float input = actual_layer->inputs_cache[k];
                actual_neuron->weight[k] += LEARNING_RATE*delta*input;
            }
            actual_neuron->bias += LEARNING_RATE*delta;
        }
    }
}

void train(Network* net, float inputs[][MAX_INPUTS], float outputs[][MAX_OUTPUTS],int num_samples, int epochs, int output_size){
    for(int i=0; i<epochs; i++){
        float total_error = 0.0;
        for (int j=0; j<num_samples; j++){
            // 1. FORWARD (Predir)
            forward_Network(net, inputs[j]);

            // 2. BACKPROPAGATION (Calcular deltes)
            backpropagation(net, outputs[j], output_size);

            // 3. UPDATE (Aprendre)
            update_weights(net);
            // Opcional: Calcular l'error quadràtic mitjà (MSE) només per visualitzar
            // (Això no afecta l'entrenament, només és per veure-ho per pantalla)
            Layer* output_layer = &net->layers[net->layer_count - 1];
            for(int k=0; k<output_size; k++){
                float diff = outputs[j][k] - output_layer->outputs_cache[k];
                total_error += diff * diff;
            }
        }

        // Imprimir l'error cada 1000 èpoques per veure el progrés
        if(i % 1000 == 0){
            printf("Epoch %d: Error mitja = %f\n", i, total_error / num_samples);
        
        }
    }
}

void normalization(float normal[], float input[][MAX_INPUTS], int input_data, int data_size){
    for(int i=0; i<data_size; i++){
        for(int j=0; j<input_data;j++){
            input[i][j] = input[i][j] / normal[j];
        }
    }
}
void read(float inputs[][MAX_INPUTS], float outputs[][MAX_INPUTS], int *samples_read, int *inputs_count, int *outputs_count){
    char filename[MAX_LETTERS];
    printf("File name: ");
    scanf("%s", filename);
    printf("Opening: [%s]\n", filename);
    FILE *fp = fopen(filename, "r");
    if(fp == NULL){
        printf("File doesn't exist!");
        return;
    }
    else{
        float normal_input[MAX_INPUTS], normal_output[MAX_INPUTS];
        fscanf(fp ,"%d%d%d", inputs_count, outputs_count, samples_read);
        for(int i=0; i<*inputs_count; i++){
            fscanf(fp, "%f",&normal_input[i]);
        }
        for(int i=0; i<*outputs_count; i++){
            fscanf(fp, "%f",&normal_output[i]);
        }
        for(int i=0; i<*samples_read;i++){
            for(int j=0; j<*inputs_count;j++){
                fscanf(fp,"%f",&inputs[i][j]);
            }
            for(int j=0; j<*outputs_count;j++){
                fscanf(fp,"%f",&outputs[i][j]);
            }
        }
        fclose(fp);
        normalization(normal_input,inputs,*inputs_count,*samples_read);
        normalization(normal_output,outputs,*outputs_count,*samples_read);
        return;   
    }
}

void save(Network net){
    char name[MAX_LETTERS];
    printf("Name to save the trained network: ");
    scanf("%s", name);
    FILE *fp = fopen(name, "w");
    if(fp == NULL){
        printf("File doesn't exist!");
        net.layer_count = 0;
        return;
    }
    fprintf(fp, "%d ", net.layer_count);
    for(int i=0; i<net.layer_count; i++){
        fprintf(fp, "%d ", net.layers[i].activation_type);
        fprintf(fp, "%d ", net.layers[i].neuron_count);
        fprintf(fp, "%d ", net.layers[i].inputs_per_neuron);
        for(int j = 0; j<net.layers[i].neuron_count; j++){
            fprintf(fp, "%f ", net.layers[i].deltas_cache[j]);
        }
        for(int j = 0; j<net.layers[i].inputs_per_neuron; j++){
            fprintf(fp, "%f ", net.layers[i].inputs_cache[j]);
        }
        for(int j = 0; j<net.layers[i].neuron_count; j++){
            fprintf(fp, "%f ", net.layers[i].outputs_cache[j]);
        }
        for(int j = 0; j<net.layers[i].neuron_count; j++){
            fprintf(fp, "%f ", net.layers[i].neurons[j].bias);
            fprintf(fp, "%f ", net.layers[i].neurons[j].delta);
            fprintf(fp, "%d ", net.layers[i].neurons[j].num_weights);
            fprintf(fp, "%f ", net.layers[i].neurons[j].output_prereulu);
            fprintf(fp, "%f ", net.layers[i].neurons[j].output);
            for(int k = 0; k<net.layers[i].neurons[j].num_weights; k++){
                fprintf(fp,"%f ", net.layers[i].neurons[j].weight[k]);
            }
        }

    }
}

Network read_network(){
    Network net;
    char name[MAX_LETTERS];
    printf("Name to take the trained network: ");
    scanf("%s", name);
    FILE *fp = fopen(name, "r");
    if(fp == NULL){
        printf("File doesn't exist\n");
        net.layer_count = 0;
        return net;
    }
    //
    fscanf(fp, "%d ", &net.layer_count);
    for(int i=0; i<net.layer_count; i++){
        fscanf(fp, "%d ", &net.layers[i].activation_type);
        fscanf(fp, "%d ", &net.layers[i].neuron_count);
        fscanf(fp, "%d ", &net.layers[i].inputs_per_neuron);
        for(int j = 0; j<net.layers[i].neuron_count; j++){
            fscanf(fp, "%f", &net.layers[i].deltas_cache[j]);
        }
        for(int j = 0; j<net.layers[i].inputs_per_neuron; j++){
            fscanf(fp, "%f", &net.layers[i].inputs_cache[j]);
        }
        for(int j = 0; j<net.layers[i].neuron_count; j++){
            fscanf(fp, "%f", &net.layers[i].outputs_cache[j]);
        }
        for(int j = 0; j<net.layers[i].neuron_count; j++){
            fscanf(fp, "%f ", &net.layers[i].neurons[j].bias);
            fscanf(fp, "%f ", &net.layers[i].neurons[j].delta);
            fscanf(fp, "%d ", &net.layers[i].neurons[j].num_weights);
            fscanf(fp, "%f ", &net.layers[i].neurons[j].output_prereulu);
            fscanf(fp, "%f ", &net.layers[i].neurons[j].output);
            for(int k = 0; k<net.layers[i].neurons[j].num_weights; k++){
                fscanf(fp,"%f ", &net.layers[i].neurons[j].weight[k]);
            }
        }

    }
    fclose(fp);
    return net;
}

void try_model(Network* net){
    if (net->layer_count == 0) {
        printf("Error: No network loaded or trained.\n");
        return;
    }

    int n_inputs = net->layers[0].inputs_per_neuron;
    int n_outputs = net->layers[net->layer_count - 1].neuron_count;
    
    float norm_inputs[MAX_INPUTS];
    float norm_outputs[MAX_OUTPUTS];
    float user_input[MAX_INPUTS];
    int continue_testing = 1;

    printf("\n--- PREDICTION SETUP ---\n");
    printf("The network works with normalized values.\n");
    printf("Enter the divisors for the %d inputs (the ones in the data file):\n", n_inputs);
    for(int i = 0; i < n_inputs; i++){
        printf("Max for input %d: ", i+1);
        scanf("%f", &norm_inputs[i]);
    }

    printf("Enter the multipliers for the %d outputs (the ones in the data file):\n", n_outputs);
    for(int i = 0; i < n_outputs; i++){
        printf("Max for output %d: ", i+1);
        scanf("%f", &norm_outputs[i]);
    }

    // Bucle per provar diferents valors sense sortir
    while(continue_testing){
        printf("\n--- NEW PREDICTION ---\n");
        printf("Enter the real input values:\n");
        
        for(int i = 0; i < n_inputs; i++){
            float raw_val;
            printf("Input %d: ", i+1);
            scanf("%f", &raw_val);
            // Normalitzem abans d'enviar a la xarxa
            user_input[i] = raw_val / norm_inputs[i];
        }

        // Executem la xarxa
        forward_Network(net, user_input);

        // Mostrem resultats
        Layer* output_layer = &net->layers[net->layer_count - 1];
        printf("\n--- RESULTS ---\n");
        for(int i = 0; i < n_outputs; i++){
            float raw_out = output_layer->outputs_cache[i];
            float real_val = raw_out * norm_outputs[i]; // Des-normalitzem
            printf("Output %d: %.4f (Approx. real value: %.2f)\n", i+1, raw_out, real_val);
        }

        printf("\nDo you want to test another case (1: Yes, 0: No): ");
        scanf("%d", &continue_testing);
    }
}

void menu(Network *net){
    int option = 0;
    while(option != 5){
        printf("Options: \n 1. Traning a Network \n 2. Save the Network \n 3. Take a create network \n 4. Try a Network \n 5. Exit");
        printf("\n Pick a option: ");
        if (scanf("%d", &option) != 1) {
            while (getchar() != '\n');
            printf("Error: Please enter a valid number.\n");
            continue;
        }
        if(option == 1){
            float inputs[MAX_DATA][MAX_INPUTS], outputs[MAX_DATA][MAX_INPUTS];
            int inputs_count, outputs_count, data_size;
            read(inputs, outputs, &data_size,&inputs_count,&outputs_count);
            int layer_count, neuron_count, epochs;
            printf("Number of layers wished:");
            scanf("%d", &layer_count);
            printf("Number of neurons wished:");
            scanf("%d", &neuron_count);
            printf("Epochs:");
            scanf("%d", &epochs);
            initialize_network(net, layer_count, neuron_count, inputs_count, outputs_count);
            printf("Training...");
            train(net, inputs, outputs, data_size, epochs,outputs_count);
            printf("Model trained\n");
        }
        if(option == 2){
            save(*net);
        }
        if(option == 3){
            *net = read_network(); 
            printf("Xarxa carregada correctament.\n");
        }
        if(option == 4){
            try_model(net);
        }
    }
    
}

int main(){
    Network net;
    menu(&net);
    printf("Goodbye!");
    return 0;
}
