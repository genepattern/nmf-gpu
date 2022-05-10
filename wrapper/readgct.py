import numpy as np
import cupy as cp

class NP_GCT:
    # #1.2
    # ros cols
    # name descrip sample1 sample2 ...
    # rowname1 rowdescrip1 value value ...   
    def __init__(self, filename=None, data=None, rowNames=None, rowDescrip=None, colNames=None):
        if filename:
            print(filename)

            # init from passed in stuff
            f = open(filename,'r')
            count=0
            f.readline() ##1.2
            dims = f.readline().split('\t') # rows cols
            numCols = int(dims[1])
            colNames = f.readline().strip().split('\t');
            # override comments because there are none in a gct but there can be # in a gene description
            data = cp.genfromtxt(fname=filename, delimiter="\t", skip_header=3, usecols=range(2,numCols+2), filling_values=0, comments='#####')  # change filling_valuesas req'd to fill in missing values
            self.data = data


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
        else:    
            self.data=data
            self.rownames=rowNames
            self.rowdescriptions=rowDescrip
            self.columnnames=colNames
        print("Loaded matrix of shape ", self.data.shape)
    
    def write_gct(self, file_path):
        """
        Writes the provided NP_GCT to a GCT file.
        If any of rownames, rowdescriptions or columnnames is missing write the 
        index in their place (starting with 1)
    
        :param file_path:
        :return:
        """
        np.set_printoptions(suppress=True)
        with open(file_path, 'w') as file:

            nRows = self.data.shape[0]
            nCols = self.data.shape[1]
            rowNames = self.rownames;
            rowDescriptions = self.rowdescriptions;
            colNames = self.columnnames;
        
            if not rowNames:
                rowNames = ["{:}".format(n) for n in range(1,self.data.shape[0]+1)]
        
            if not rowDescriptions:
                rowDescriptions = rowNames
        
            if not colNames:
                colNames =  ["{:}".format(n) for n in range(1,self.data.shape[1]+1)]
        
            file.write('#1.2\n' + str(nRows) + '\t' + str(nCols) + '\n')
            file.write("Name\tDescription\t")
            file.write(colNames[0])

            for j in range(1, nCols):
                file.write('\t')
                file.write(colNames[j])

            file.write('\n')

            for i in range(nRows):
                file.write(rowNames[i] + '\t')
                file.write(rowDescriptions[i] + '\t')
                file.write(str(self.data[i,0]))
                for j in range(1, nCols):
                    file.write('\t')
                    file.write(str(self.data[i,j]))

                file.write('\n')
            print("File written " + file_path)





    def __write_gct(self, file_path):
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
         
        
 
