

class lin_comb(object):



    def __init__(self,transform,data_type=float):
        self.transform = transform
    
    def op(self,data):

        return self.transform.op(data)

        


    def adj_op(self,data):

        return self.transform.adj_op(data)
    
    
    
    def op_single(self,data):

        return self.transform.op_single(data)

        


    def adj_op_single(self,data):

        return self.transform.adj_op_single(data)


        

    def get_l1norm(self):

        return self.transform.l1norm()




        
