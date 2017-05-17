def Backpropagation (self, targets, alfa, momentum):

        error = targets - self.output
        output_deltas = error * self.output
       
        self.change_linear_output = output_deltas * self.sum_output
        self.change_noLinear_output = output_deltas * self.sum_output
        
        self.output_weights[0] = self.output_weights[0] + ((alfa * self.change_linear_output) + (momentum * self.change_linear_output))
        self.output_weights[1] = self.output_weights[1] + ((alfa * self.change_noLinear_output) + momentum * self.change_noLinear_output)
  
    
        #calculando os deltas escondidos
        
        hidden_linear_deltas = []
        hidden_noLinear_deltas = []
        error_linear = 0.0
        error_noLinear = 0.0
        for j in range(len(self.linear_input_weights)):
            error_linear += output_deltas * self.output_weights[0]
            hidden_linear_deltas.append(error * self.sum_linear)
            error_noLinear += output_deltas * self.output_weights[0]
            hidden_noLinear_deltas.append(error * self.sum_noLinear) 
        
        #atualizando os pesos escondidos
        for j in range (len(self.linear_input_weights)):
            self.change_linear_input = hidden_linear_deltas[j] * self.sum_linear
            self.change_noLinear_input = hidden_noLinear_deltas[j] * self.sum_noLinear
            # print 'activation',self.activate_initial[i],'synapse',i,j,'change',change
            self.linear_input_weights[j] += alfa * self.change_linear_input + momentum * self.change_linear_input
            self.noLinear_input_weights[j] += alfa * self.change_noLinear_input + momentum * self.change_noLinear_input
                