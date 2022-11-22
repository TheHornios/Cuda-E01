/**
* ARQUITECTURA DE COMPUTADORES
* 2º Grado en Ingenieria Informatica
*
* Entrega 1 
*
* Alumno: Rodrigo Pascual Arnaiz y Villar Solla, Alejandro
* Fecha: 02/11/2022
*
*/

///////////////////////////////////////////////////////////////////////////
// includes
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
///////////////////////////////////////////////////////////////////////////
// defines
#define M 7
#define N 25
///////////////////////////////////////////////////////////////////////////
// declaracion de funciones
// HOST: funcion llamada desde el host y ejecutada en el host
/**
* Funcion: propiedadesDispositivo
* Objetivo: Mustra las propiedades del dispositvo, esta funcion
* es ejecutada llamada y ejecutada desde el host
*
* Param: INT id_dispositivo -> ID del dispotivo
* Return: cudaDeviceProp -> retorna el onjeto que tiene todas las
* propiedades del dispositivo CUDA
*/
__host__ void propiedadesDispositivo(int id_dispositivo)
{
	cudaDeviceProp deviceProp;
	cudaGetDeviceProperties(&deviceProp, id_dispositivo);
	// calculo del numero de cores (SP)
	int cuda_cores = 0;
	int multi_processor_count = deviceProp.multiProcessorCount;
	int major = deviceProp.major;
	int minor = deviceProp.minor;
	char* arquitectura = (char*)"";
	switch (major)
	{
	case 1:
		//TESLA
		cuda_cores = 8;
		arquitectura = (char*)"TESLA";
		break;
	case 2:
		//FERMI
		arquitectura = (char*)"FERMI";
		if (minor == 0)
			cuda_cores = 32;
		else
			cuda_cores = 48;
		break;
	case 3:
		//KEPLER
		arquitectura = (char*)"KEPLER";
		cuda_cores = 192;
		break;
	case 5:
		//MAXWELL
		arquitectura = (char*)"MAXWELL";
		cuda_cores = 128;
		break;
	case 6:
		//PASCAL
		arquitectura = (char*)"PASCAL";
		cuda_cores = 64;
		break;
	case 7:
		//VOLTA
		arquitectura = (char*)"VOLTA";
		cuda_cores = 64;
		break;
	case 8:
		//AMPERE
		arquitectura = (char*)"AMPERE";
		cuda_cores = 128;
		break;
	default:
		arquitectura = (char*)"DESCONOCIDA";
		//DESCONOCIDA
		cuda_cores = 0;
		printf("!!!!!dispositivo desconocido!!!!!\n");
	}
	int rtV;
	cudaRuntimeGetVersion(&rtV);
	// presentacion de propiedades
	printf("***************************************************\n");
	printf("DEVICE %d: %s\n", id_dispositivo, deviceProp.name);
	printf("***************************************************\n");
	printf("> CUDA Toolkit \t\t\t\t: %d.%d\n", rtV / 1000, (rtV % 1000) / 10);
	printf("> Capacidad de Computo \t\t\t: %d.%d\n", major, minor);
	printf("> Arquitectura CUDA \t\t\t: %s \n", arquitectura);
	printf("> No. de MultiProcesadores \t\t: %d \n",
		multi_processor_count);
	printf("> No. de CUDA Cores (%dx%d) \t\t: %d \n", cuda_cores,
		multi_processor_count, cuda_cores *
		multi_processor_count);
	printf("> No. max. de Hilos (por bloque) \t: %d \n",
		deviceProp.maxThreadsPerBlock);
	printf("> Memoria Global (total) \t\t: %zu MiB\n",
		deviceProp.totalGlobalMem / (1 << 20));

	printf("***************************************************\n");
	printf("> KERNEL DE %i BLOQUE con %i HILOS:\n", 1, N * M);
	printf("\teje x -> %i hilos\n", N);
	printf("\teje y -> %i hilos\n", M);
	
}

///////////////////////////////////////////////////////////////////////////
// HOST: funcion llamada desde el host y ejecutada en el host
/**
* Funcion: rellenarVectorHst
* Objetivo: Funcion que rellena un array pasado por parametro
* con numero aleatorios del 1 al 9
*
* Param: INT* arr -> Puntero del array a rellenar
* Return: void
*/
__host__ void rellenarVectorHst(int* arr)
{

	

	for (size_t i = 0; i < M; i++)
	{
		int num_aleatorio = rand() % 10;
		for (size_t t = 0; t < N; t++)
		{

			arr[N * i + t] = num_aleatorio;
		}
	}
}

///////////////////////////////////////////////////////////////////////////
// KERNEL: Función que deja las columnas impares a 0
/**
* Funcion: desplazarAbajo
* Objetivo: Funcion que desplaza una una fila completa de un array una posicion mas abajo 
*  y mueve el ultimo elemento a la primera posicion
* 
* Param: INT* arr -> Puntero del array que tiene los datos
* Param: INT* arr_final -> Puntero del array a rellenar
* Return: void
*/

__global__ void desplazarAbajo(int* arr, int* arr_final )
{
	int columna = threadIdx.x;
	int fila = threadIdx.y;
	int pos = fila * N + columna;


	if (fila == 0) {
		arr_final[pos] = arr[( M - 1 ) * N + columna];
	}
	else {
		arr_final[pos] = arr[( fila - 1 )* N + columna];
	}

}
///////////////////////////////////////////////////////////////////////////
// MAIN: rutina principal ejecutada en el host
int main(int argc, char** argv)
{
	srand(time(NULL));
	// buscando dispositivos
	int numero_dispositivos;
	cudaGetDeviceCount(&numero_dispositivos);
	if (numero_dispositivos != 0)
	{
		for (int i = 0; i < numero_dispositivos; i++)
		{
			propiedadesDispositivo(i);
		}
	}
	else
	{
		printf("!!!!!ERROR!!!!!\n");
		printf("Este ordenador no tiene dispositivo de ejecucion CUDA\n");
		printf("<pulsa [INTRO] para finalizar>");
		getchar();
		return 1;
	}

	// Básico 5
	// Declaración de variables
	int* hst_original, * hst_final;
	int* dev_original, * dev_final;
	// Declaración de eventos
	cudaEvent_t start;
	cudaEvent_t stop;
	// Asignación de espacio a las variables en el host
	hst_original = (int*)malloc(N * M * sizeof(int));
	hst_final = (int*)malloc(N * M * sizeof(int));

	// Asignación de espacio a las variables en el device
	cudaMalloc((void**)&dev_original, N * M * sizeof(int));
	cudaMalloc((void**)&dev_final, N * M * sizeof(int));
	// Creación de eventos
	cudaEventCreate(&start);
	cudaEventCreate(&stop);
	// Rellenar con información
	

	rellenarVectorHst(hst_original );

	// Copiar datos al dispositivo
	cudaMemcpy(dev_original, hst_original, sizeof(int) * N * M,
		cudaMemcpyHostToDevice);
	// Desplazar filas
	dim3 blocks(1);
	dim3 threads(N, M);
	//// Marca de inicio
	cudaEventRecord(start, 0);
	//// Función KERNEL
	desplazarAbajo <<<blocks, threads >>> (dev_original, dev_final);
	//// Marca de fin
	cudaEventRecord(stop, 0);
	//// Sincronizar Eventos
	cudaEventSynchronize(stop);
	// Traer datos del device
	cudaMemcpy(hst_final, dev_final, sizeof(int) * N * M,
		cudaMemcpyDeviceToHost);
	// Mostrar tiempo de ejecución, original y resultado

	//// Calcular tiempo
	float elapsedTime;
	cudaEventElapsedTime(&elapsedTime, start, stop);
	printf("> Tiempo de ejecucion: %0.6f ms\n", elapsedTime);
	printf("> MATRIZ ORIGINAL:\n");
	for (int y = 0; y < M; y++)
	{
		for (int x = 0; x < N; x++)
		{
			printf("%i  ", hst_original[N * y + x]);
		}
		printf("\n");
	}

	printf("\n");
	printf("> MATRIZ FINAL:\n");
	for (int y = 0; y < M; y++) 
	{
		for (int x = 0; x < N; x++)
		{
			printf("%i  ", hst_final[N * y + x]);
		}
		printf("\n");
	}
	// Salida del programa
	time_t fecha;
	time(&fecha);
	printf("***************************************************\n");
	printf("Programa ejecutado el: %s\n", ctime(&fecha));
	printf("<pulsa [INTRO] para finalizar>");
	getchar();
	return 0;
}
///////////////////////////////////////////////////////////////////////////