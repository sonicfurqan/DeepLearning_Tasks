# TENSOR

A tensor can be
1.Single number(Scalar)
2.Array List (1d tensor)
3.Matrix (2d tensor)
....and so on

## Tensor are imutable object i.e data in it cannot be changed once created

## tf.tensor (values, shape?, dtype?)

    generic method to create tensor

    @param
    values = list of array or arrray of arrays
    shape(Optional) = what shape the list  is to be converted to
    example
     a.if you want array to be convertd to 2*3 matrix. Shape would be [2,3] where 2 is row and 3 is coloum
     b.if you want array to be converted to three 2*2 matrix's. Shape would be [3,2,2] where 3 is number of matrix  and 2,2 is row and coloum respectivly
    dtype(Optional) = array can be of followinf data types 'float32'|'int32'|'bool'|'complex64'|'string'

## tf.ones (shape, dtype?)

    create a tensor matrix that contains only 1
    @param
    shape = size of matrix that is to be created and number of matrix
    dtype(Optional) = array can be of following data types 'float32'|'int32'|'bool'|'complex64'|'string'

# VARIABLE

variables in tensorflow are mutable objects i.e you can update the data in it. Variable can be created using exsiting tensor

## tf.variable (initialValue, trainable?, name?, dtype?)

Generic method to create tensor variable

    @param
    initialValue = list of array or arrray of arrays
    trainable(Optional) = optimizers are allowed to update it
    name (Optional)= Name of the variable
    dtype(Optional) = array can be of followinf data types 'float32'|'int32'|'bool'|'complex64'|'string'
