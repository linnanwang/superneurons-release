#include <comm_engine.h>

namespace SuperNeurons{

template <class value_type>
int comm_engine_mpi_t<value_type>::bcast(void *buffer, int count, int root)
{
	switch (sizeof(value_type))
  	{
		case 2 :
			FatalError("Unsupported data type");
			break;
		case 4 :
			MPI_Bcast(buffer, count, MPI_FLOAT, root, comm);
			break;
		case 8 :
			MPI_Bcast(buffer, count, MPI_DOUBLE, root, comm);
			break;
		default : FatalError("Unsupported data type");
   }
}

template <class value_type>
int comm_engine_mpi_t<value_type>::reduce(void *sendbuf, void *recvbuf, int count, int root)
{
	if (rank == root) {
		switch (sizeof(value_type))
	    {
	        case 2 :
	            FatalError("Unsupported data type");
	            break;
	        case 4 :
	            MPI_Reduce(MPI_IN_PLACE, recvbuf, count, MPI_FLOAT, MPI_SUM, root, comm);
	            break;
	        case 8 :
	            MPI_Reduce(MPI_IN_PLACE, recvbuf, count, MPI_DOUBLE, MPI_SUM, root, comm);
	            break;
	        default : FatalError("Unsupported data type");
	    }
	} else {
        switch (sizeof(value_type))
        {
            case 2 :
                FatalError("Unsupported data type");
                break;
            case 4 :
                MPI_Reduce(sendbuf, NULL, count, MPI_FLOAT, MPI_SUM, root, comm);
                break;
            case 8 :
                MPI_Reduce(sendbuf, NULL, count, MPI_DOUBLE, MPI_SUM, root, comm);
                break;
            default : FatalError("Unsupported data type");
        }
	}
}