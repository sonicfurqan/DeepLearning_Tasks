window.addEventListener('DOMContentLoaded', init());
function init() {}

/*A tensor can be
 * 1.Single number(Scalar)
 * 2.Array List (1d tensor)
 * 3.Matrix (2d tensor)
 * ....and so on
 */
function tensor() {
  /*@Syntax tf.tensor (values, shape?, dtype?)
   * @patam
   * values = list of array or arrray of arrays
   * shape(Optional) = what shape the list  is to be converted to
   * example
   *  a.if you want array to be convertd to 2*3 matrix. Shape would be [2,3] where 2 is row and 3 is coloum
   *  b.if you want array to be converted to three 2*2 matrix's. Shape would be [3,2,2] where 3 is number of matrix  and 2,2 is row and coloum respectivly
   * dtype(Optional) = array can be of followinf data types 'float32'|'int32'|'bool'|'complex64'|'string'
   */
  const tensor_1d = tf.tensor([1, 2, 3, 4]);
  //creates a 1d array tensor
  debug('1d=' + tensor_1d.toString());
  const tensor_1d_2arrays = tf.tensor([1, 2, 3, 4], [2, 1, 2]);
  //creates two 1d array tensor
  debug('1d*2=' + tensor_1d_2arrays.toString());
  const tensor_2d = tf.tensor([1, 2, 3, 4], [2, 2]);
  //creates a 2d array tensor
  debug('2d=' + tensor_2d.toString());
  const tensor_2d_2arrays = tf.tensor([1, 2, 3, 4, 5, 6, 7, 8], [2, 2, 2]);
  //creates two 2d array tensor
  debug('2d*2=' + tensor_2d_2arrays.toString());
}

function create_ones_tensor() {
  /*@Syntax  tf.ones (shape, dtype?)
   * create a tensor matrix that contains only 1
   * @patam
   * shape = size of matrix that is to be created and number of matrix
   * dtype(Optional) = array can be of following data types 'float32'|'int32'|'bool'|'complex64'|'string'
   */
  const tensor2d = tf.ones([2, 2]);
  debug(tensor2d.toString());
}

function create_tows_tensor() {
  /*@Syntax  tf.ones (shape, dtype?)
   * create a tensor matrix that contains only 1
   * @patam
   * shape = size of matrix that is to be created and number of matrix
   * dtype(Optional) = array can be of following data types 'float32'|'int32'|'bool'|'complex64'|'string'
   */
  const tensor2d = tf.zeros([2, 2]);
  debug(tensor2d.toString());
}

function tensor_specfic() {
  /*@Syntax tf.scalar (value, dtype?)
   * creates a tensor of single value
   * @patam
   * values = list of array or arrray of arrays
   * dtype(Optional) = array can be of followinf data types 'float32'|'int32'|'bool'|'complex64'|'string'
   */
  const scalar = tf.scalar(3.14, 'int32');
  debug('scalar : ' + scalar.toString());
  /*@Syntaxtf.tensor1d (values, dtype?)
   * creates a tensor of array list
   * @patam
   * values = list of array or arrray of arrays
   * dtype(Optional) = array can be of followinf data types 'float32'|'int32'|'bool'|'complex64'|'string'
   */
  const tensor1d = tf.tensor1d([1, 2, 3]);
  debug('tensor 1d : ' + tensor1d.toString());
  /*@Syntaxtf.tensor2d (values, dtype?)
   * creates a tensor of matrix type
   * @patam
   * values = list of array or arrray of arrays
   * shape = define number of matrix and shape of matrix
   * dtype(Optional) = array can be of followinf data types 'float32'|'int32'|'bool'|'complex64'|'string'
   */
  const tensor2d = tf.tensor2d([1, 2, 3, 4], [2, 2]);
  debug('tensor 2d : ' + tensor2d.toString());
}

function clean() {
  /* */
}

function debug(msg) {
  const dom = document.querySelector('.output');
  dom.innerHTML += msg + '<br/>';
}
