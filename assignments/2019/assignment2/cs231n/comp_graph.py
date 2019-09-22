"""
Simple computation graph implmentation. Doesn't handle
cycles.
Used to do back propigation for a batch normalization node.
"""


from collections import deque
import numpy as np

class CompNode(object):
    def __init__(self):
        self.name = "unamed"            
        self.output = None

    def cprint(self, *args):
        #print(args)
        pass      

class PlusNode(CompNode):
    def __init__(self):
        CompNode.__init__(self)
        self.name = "plus" 

    def execute(self, inputs):
        self.cprint("Executing:", self.name, inputs)
        self.output = inputs[0] + inputs[1]

    def reverse(self, dout, output_map):
        return [dout, dout]

class SubNode(CompNode):
    def __init__(self):
        CompNode.__init__(self)
        self.name = "minus" 

    def execute(self, inputs):
        self.cprint("Executing:", self.name, inputs)
        self.output = inputs[0]  - inputs[1]

    def reverse(self, dout, output_map):
        return [dout, -dout]

class SqrtNode(CompNode):
    def __init__(self):
        CompNode.__init__(self)
        self.name = "sqrt" 

    def execute(self, inputs):
        self.cprint("Executing:", self.name, inputs)
        self.output = np.sqrt(inputs[0])
        self.cprint("Sqrt out", self.output)

    def reverse(self, dout, output_map):
        out = dout / (2 *self.output)
        return [out]

class DivNoode(CompNode):
    def __init__(self):
        self.name = "divide"            

    def execute(self, inputs):
        self.cprint("Executing:", self.name, inputs)
        self.numerator = inputs[0]
        self.denominator = inputs[1]
        self.output = inputs[0]  / inputs[1]
        self.cprint("divout", self.output)

    def reverse(self, dout, output_map):
        out0 = dout / self.denominator
        out1 = -dout * self.numerator / (self.denominator * self.denominator)
        return [out0, out1]

class InputNode(CompNode):
    def __init__(self):
        self.name = "input"            
        pass

    def execute(self, inputs):
        self.cprint("Executing:", self.name, inputs)
        self.output = inputs[0]

    def reverse(self, dout, output_map):

        if 'dx' in output_map:
            #print("dx - adding")
            #print("existing: ", output_map['dx'])
            #print("incoming:", dout)
            output_map['dx'] += dout
        else:
            output_map['dx'] = dout

class ConstantNode(CompNode):
    def __init__(self, value):
        self.name = "constant"            
        self.value = value
        pass

    def execute(self, inputs):
        self.cprint("Executing:", self.name, inputs)
        self.output = self.value

    def reverse(self, dout, output_map):
        return [np.zeros_like(dout)]        

class MuNode(CompNode):
    def __init__(self):
        self.name = "average" 
           
        pass

    def execute(self, inputs):
        self.cprint("Executing:", self.name, inputs)
        self.output = np.mean(inputs[0], axis = 0)
        self.shape = inputs[0].shape
        self.cprint("mu", self.output)

    def reverse(self, dout, output_map):
        vsum = np.sum(dout, axis = 0)

        # by convention, we want an N x D output. Adding vsum
        # to an array of 0's will replicate the rows.
        out = (vsum + np.zeros(self.shape)) /self.shape[0]
        return [out]

class SigmaNode(CompNode):
    def __init__(self):
        self.name = "sigma"            
        pass

    def execute(self, inputs):
        self.cprint("Executing:", self.name, inputs)
        self.x = inputs[0]
        out = np.var(self.x, axis = 0)
        out = np.zeros_like(self.x) + out
        self.output = out
        self.cprint("sigma", self.output)

    def reverse(self, dout, output_map):
        vsum = np.sum(dout, axis = 0)

        xsum = np.sum(self.x, axis = 0)

        n = self.x.shape[0]

        xt = 2*(self.x/n  - xsum/(n*n))

        out = xt * vsum
        return [out]         

    def _execute(self, inputs):
        self.cprint("Executing:", self.name, inputs)
        self.output = np.ones_like(inputs[0])
        self.cprint("sigma", self.output)        

    def _reverse(self, dout, output_map):        
        return [np.zeros_like(dout)]


class NodeScheduler(object):
    def __init__(self):
        self.parent_map = {}
        self.rev_order = None
        pass

    def add_node(self, node, parents = None):
        assert(node not in self.parent_map)
        self.parent_map[node] = parents

    def find_root(self):
        parents = set()
        nodes = set()
        for n,v in self.parent_map.items():
            nodes.add(n)
            if v:
                parents.update(v)

        diff = nodes - parents
        root = diff.pop()
        assert(not diff)   # Should be only one root
        return root

    def calc_rev_order(self):
        order = []

        d = deque()
        d.append( (self.find_root(), 0, None))

        while d:
            node, idx, child  = d.popleft()
            order.append((node, idx, child))
            parents = self.parent_map[node]
            if parents:
                for i, p in enumerate(parents):
                    d.append((p, i, node))

        return order

    def forward(self, input):
        self.rev_order = self.calc_rev_order()
        
        order = self.rev_order.copy()
        order.reverse()

        for node, _, _ in order:

            parents = self.parent_map[node]

            if parents is None:
                node.execute([input])
            else:
                inputs = [parent.output for parent in parents]
                node.execute(inputs)

        return order[-1][0].output  

    def reverse(self, out):
        output_map = {}
        reverse_map = {}

        for node, idx, child in self.rev_order:
            if child is None:
                child_out = out
                #print(node.name, idx, "<no child>")
            else:
                #print("Reverse map", child.name, idx, reverse_map[child])
                child_out = (reverse_map[child])[idx]
                #print(node.name, idx, child.name)
            
            reverse_output = node.reverse(child_out, output_map)
            #print("child_out", child_out)
            #print("reverse_output", reverse_output)
            reverse_map[node] = reverse_output

        return output_map            



def create_batch_norm(s, epsilon_value):
    xinput_a = InputNode()
    xinput_a.name = "Input to numerator"

    xinput_b = InputNode()
    xinput_b.name = "Input to Mu numerator"

    mu = MuNode()
    mu.name = "Mu numerator"

    numerator = SubNode()

    epsilon = ConstantNode(epsilon_value)
    epsilon.name = "Epsilon constant"
    sigma = SigmaNode()

    splus = PlusNode()

    xinput_c = InputNode()
    xinput_c.name = "Input to Sigma"

    denominatior = SqrtNode()

    div = DivNoode()
    
    s.add_node(div, [numerator, denominatior])
    s.add_node(numerator, [xinput_a, mu])
    s.add_node(xinput_a)

    s.add_node(mu, [xinput_b])
    s.add_node(xinput_b)

    s.add_node(denominatior, [splus])
    s.add_node(splus, [sigma, epsilon])
    s.add_node(sigma, [xinput_c])
    s.add_node(xinput_c)
    s.add_node(epsilon)
    
    #s.add_node(sigma, [xinput_c])
    #s.add_node(xinput_c)

    return div

class batch_norm_graph(object):
    def __init__(self, epsilon):
        self.scheduler = NodeScheduler()
        self.root = create_batch_norm(self.scheduler, epsilon)
        pass    

    def forward(self, x):
        return self.scheduler.forward(x)

    def reverse(self, out):
        return self.scheduler.reverse(out)



