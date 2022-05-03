import numpy as np

class NP_GCT:
    # #1.2
    # ros cols
    # name descrip sample1 sample2 ...
    # rowname1 rowdescrip1 value value ...   
    def __init__(self, filename=None, data=None, rowNames=None, rowDescrip=None, colNames=None):
        print(filename)
        if filename:
            # init from passed in stuff
            data = np.genfromtxt(fname=filename, delimiter="\t", skip_header=3, filling_values=0)  # change filling_values as req'd to fill in missing values
            self.data = data[:,2:]
            f = open(filename,'r')
            count=0
            f.readline() ##1.2
            dims = f.readline().split('\t') # rows cols
            colNames = f.readline().strip().split('\t');

            self.columnnames = colNames[2:];
            self.rownames = [None] * int(dims[0])
            self.rowdescriptions = [None] * int(dims[0])

            while True:
                # Get next line from file
                line = f.readline()

                # if line is empty
                # end of file is reached
                if not line:
                    break

                line = line.split('\t');
                name = line[0]
                description=line[1]
                self.rownames[count] = name
                self.rowdescriptions[count] = description
                count += 1

            f.close()
            print("loaded from file")
        else:    
            print("loaded from vars")
            self.data=data
            self.rownames=rowNames
            self.rowdescriptions=rowDescrip
            self.columnnames=colNames
    
    def write_gct(self, file_path):
        """
        Writes the provided DataFrame to a GCT file.
        Assumes that the DataFrame matches the structure of those produced
        by the GCT() function in this library
        :param df:
        :param file_path:
        :return:
        """
        np.set_printoptions(suppress=True)
        with open(file_path, 'w') as file:
            
            file.write('#1.2\n' + str(len(self.rownames)) + '\t' + str(len(self.columnnames)) + '\n')
            file.write("Name\tDescription\t")
            file.write(self.columnnames[0])
            
            for j in range(1, len(self.columnnames)):
                file.write('\t')
                file.write(self.columnnames[j])
                
            file.write('\n')
                
            for i in range(len(self.rownames)):
                file.write(self.rownames[i] + '\t')
                file.write(self.rowdescriptions[i] + '\t')
                file.write(str(self.data[i,0]))
                for j in range(1, len(self.columnnames)):
                    file.write('\t')
                    file.write(str(self.data[i,j]))
                    
                file.write('\n')
        print("File written " + file_path)
         
        
 
