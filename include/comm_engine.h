#if !defined(_COMM_ENGINE_H_)
#define _COMM_ENGINE_H_

#include <vector>
#include <registry.h>
#include <util/common.h>
#include <util/error_util.h>
#include <layer/base_layer.h>
#include <util/superneurons_math.h>

#include <mpi.h>

namespace SuperNeurons{

class comm_engine_t
{
public:
	virtual int bcast(void *buffer, int count, int root) = 0;
	virtual int reduce(void *sendbuf, void *recvbuf, int count, int root) = 0;
};

template <class value_type>
class comm_engine_mpi_t: public comm_engine_t
{
private:
	MPI_Comm comm;
	int size;
	int rank;
	int nb_gpus_per_node;
public:
	comm_engine_mpi_t(int ngpn)
	{
		MPI_Init(NULL, NULL);
		comm = MPI_COMM_WORLD;
		nb_gpus_per_node = ngpn;
		MPI_Comm_size(MPI_COMM_WORLD, &size);
		MPI_Comm_rank(MPI_COMM_WORLD, &rank);
		cudaSetDevice(rank % nb_gpus_per_node);
	};
	~comm_engine_mpi_t()
	{
		MPI_Finalize();
	};
	int bcast(void *buffer, int count, int root);
	int reduce(void *sendbuf, void *recvbuf, int count, int root);
};

}

#endif // _COMM_ENGINE_H_