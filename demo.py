from petsc4py import PETSc
rank = PETSc.COMM_WORLD.getRank()
num_ranks = PETSc.COMM_WORLD.getSize()

x = PETSc.Vec().createMPI(4) # VecCreateMPI: Creates a parallel vector.  size=4
x.setValues([0,1,2,3], [10,20,30,40]) # VecSetValues: Inserts or adds values into certain locations of a vector.  x[0]=10, x[1]=20, x[2]=30, x[3]=40

print ('Rank',rank,'has this portion of the MPI vector:', x.getArray() ) # VecGetArray: Returns a pointer to a contiguous array that contains this processor's portion of the vector data.

vec_sum = x.sum() # VecSum: Computes the sum of all the components of a vector. 10+20+30+40=100

if rank == 0:
    print ('Sum of all elements of vector x is',vec_sum,'and was computed using',num_ranks,'MPI processes.')
