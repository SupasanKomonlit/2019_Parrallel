// FILE			: stram_matrix_multi.c
// AUTHOR		: K.Supasan
// CREATE ON	: 2019, October 30 (UTC+0)
// MAINTAINER	: K.Supasan

// MACRO DETAIL

// README
//  Solve solution matrix_c = matrix_a * matrix_b

// REFERENCE

// MACRO SET

// MACRO CONDITION

#include    <stdlib.h>
#include    <stdio.h>
#include    <math.h>
#include    <assert.h>

typedef int data_type;

__global__
void kernalMulMatrixTranspose( data_type* matrix_a,
        data_type* matrix_b_t,
        data_type* matrix_c,
        unsigned int same_size,
        unsigned int offset_row_index,
        unsigned int offset_column_index )
{
    data_type result = 0;
    // I will design row_index , column_index that is tile index
    //  Tile will have dimension 32 * 32 only
    unsigned int a_index = offset_row_index + threadIdx.y*same_size;
    unsigned int b_index = offset_column_index + threadIdx.x*same_size;
    for( unsigned int run = 0 ; run < same_size ; run++ )
    {
        result += matrix_a[ a_index + run ] * matrix_b_t[ b_index + run ];
    } 
    // printf( "For (row,column) : (%4d,%4d) value is %d\n", a_index , b_index , result );
    matrix_c[ ( offset_row_index + threadIdx.y ) * same_size + offset_column_index + threadIdx.x ] = result;
} // kernalMulMatrixTranspose

void mulMatrixTranspose( data_type* matrix_a, 
        data_type* matrix_b_t, 
        data_type* matrix_c,
        unsigned int row_size,
        unsigned int same_size,
        unsigned int column_size )
{
    for( unsigned int column = 0 ; column < column_size ; column++ )
    {
        unsigned int index_column = column * same_size;
        for( unsigned int row = 0 ; row < row_size ; row++ )
        {
            unsigned int index_row = row * same_size;
            data_type result_value = 0 ;
            for( unsigned int run = 0 ; run < same_size ; run++ )
            {
                result_value += matrix_b_t[ index_column + run ] * matrix_a[ index_row + run ];
            }
//            unsigned int result_index = index_row * column_size + index_column;
//            printf( "Value %d for position is %d\n" , result_value , result_index );
            matrix_c[ row * column_size + column] = result_value;
        }
    }
} // mulMatrixTranspose

void transposeMatrix( data_type* origin_matrix,
        data_type* transpose_matrix,
        unsigned int row_size,
        unsigned int column_size );

void initMatrix( data_type* ptr_matrix, 
        unsigned int row_size, 
        unsigned int column_size, 
        data_type value );

void randomMatrix( data_type* ptr_matrix , unsigned int row_size , unsigned int column_size );

void printMatrix( data_type* ptr_matrix , unsigned int row_size , unsigned int column_size );

int main()
{
    // Assum size of matrix m * n is row * column
    unsigned int column_a = 1024;
    unsigned int row_a = 1024;
    unsigned int column_b = 1024;
    unsigned int row_b = 1024;
    unsigned int column_c = column_a;
    unsigned int row_c = row_b;
    
    data_type *ptr_matrix_a;
    data_type *ptr_matrix_b;
    data_type *ptr_matrix_b_t;
    data_type *ptr_matrix_c;

/*
    ptr_matrix_a = ( data_type* ) malloc( column_a * row_a * sizeof( data_type ) );
    ptr_matrix_b = ( data_type* ) malloc( column_b * row_b * sizeof( data_type ) );    
    ptr_matrix_c = ( data_type* ) malloc( column_c * row_c * sizeof( data_type ) );
    ptr_matrix_b_t = ( data_type* ) malloc( column_b * row_b * sizeof( data_type ) );
*/
    // change to malloc by use cuda
    
    data_type* d_matrix_a;
    data_type* d_matrix_b_t;
    data_type* d_matrix_c;

    cudaMallocHost( &ptr_matrix_a, column_a * row_a * sizeof( data_type ), cudaHostAllocDefault );
    cudaMallocHost( &ptr_matrix_b_t, column_b * row_b * sizeof( data_type ), cudaHostAllocDefault );
    cudaMallocHost( &ptr_matrix_b, column_b * row_b * sizeof( data_type ), cudaHostAllocDefault );
    cudaMallocHost( &ptr_matrix_c, column_c * row_c * sizeof( data_type ), cudaHostAllocDefault );


    cudaMalloc( &d_matrix_a , column_a * row_a * sizeof( data_type ) );
    cudaMalloc( &d_matrix_b_t , column_b * row_b * sizeof( data_type ) );
    cudaMalloc( &d_matrix_c , column_c * row_c * sizeof( data_type ) );
     

    // Assume Dimension more than 1024 for all dimension
    dim3 block_dimension,grid_dimension;
    block_dimension = dim3( 32 , 32 );
    grid_dimension = dim3( row_a / 32 , column_b / 32 );

    randomMatrix( ptr_matrix_a, row_a , column_a );
    randomMatrix( ptr_matrix_b, row_b , column_b );
    initMatrix( ptr_matrix_c , row_c , column_c , 0 );
    transposeMatrix( ptr_matrix_b , ptr_matrix_b_t , row_b , column_b );
    /*
    printf("Matrix_a\n");    
    printMatrix( ptr_matrix_a , row_a , column_a );
    printf("Matrix_B\n");    
    printMatrix( ptr_matrix_b , row_b , column_b );
    */
    // CPU process
    /*
    mulMatrixTranspose( ptr_matrix_a , ptr_matrix_b_t , ptr_matrix_c ,
            row_a , column_a , column_b );
    printf("Matrix_c by CPU\n");
    printMatrix( ptr_matrix_c , row_a , column_b );
    */
    

    // Create Event for manage about process
    cudaEvent_t event_copy_matrix_b_t[ column_b / 32 ];
    cudaEvent_t event_copy_matrix_a[ row_a / 32 ];
    cudaEvent_t event_copy_matrix_c[ row_a / 32 ];
    for( unsigned int run = 0 ; run < row_a / 32 ; run++ )
    { 
        cudaEventCreate( &event_copy_matrix_a[run] );
        cudaEventCreate( &event_copy_matrix_c[ run ] );
    }
    for( unsigned int run = 0 ; run < column_b / 32 ; run++ )
    {
        cudaEventCreate( &event_copy_matrix_b_t[ run ] );
    }

    // size of copy b transpose will copy 32 row but 
    //  each row of transpose have size member equal row size of origin matrix B
    unsigned int size_copy_b_t = sizeof( data_type ) * row_b * 32;
    // size of copy a will copy 32 row for calculate
    unsigned int size_copy_a = sizeof( data_type ) * column_a * 32;
    // size of copy C we will copy 32 row and column c equal column b
    unsigned int size_copy_c = sizeof( data_type ) * column_b * 32;
    // Below variable is stream line use to manage process
    cudaStream_t stream_copy_a, stream_copy_b, stream_copy_c, stream_calculate;
    cudaStreamCreate( &stream_copy_a );
    cudaStreamCreate( &stream_copy_b );
    cudaStreamCreate( &stream_copy_c );
    cudaStreamCreate( &stream_calculate );
    // Below variable about recoed time to run all process
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord( start  , stream_copy_c ); 
    for( unsigned int run_a = 0 ; run_a < grid_dimension.x ; run_a++ )
    {
        // copy matrix a some part
        cudaMemcpyAsync( &d_matrix_a[ run_a * size_copy_a ], 
                &ptr_matrix_a[ run_a * size_copy_a ],
                size_copy_a,
                cudaMemcpyHostToDevice,
                stream_copy_a );
        // record event copy matrix a
        cudaEventRecord( event_copy_matrix_a[ run_a ] , stream_copy_a );
        for( unsigned int run_b = 0 ; run_b < grid_dimension.y ; run_b++ )
        {
            if( run_a == 0 ) // Mean you never copy value b
            {
                // copy matrix_b some part for first time
                cudaMemcpyAsync( &d_matrix_b_t[ run_b * size_copy_b_t ],
                        &ptr_matrix_b_t[ run_b * size_copy_b_t ],
                        size_copy_b_t,
                        cudaMemcpyHostToDevice,
                        stream_copy_b );
                // record event copy matrix b
                cudaEventRecord( event_copy_matrix_b_t[ run_b ] , stream_copy_b );
                // stream_calculate must to wait copy matrix b for the first time
                cudaStreamWaitEvent( stream_calculate , event_copy_matrix_b_t[ run_b ] , 0 );     
            }
            if( run_b == 0 ) // beforce calculate you must wait copy a finish
            {
                // stream_calculate first process muset wait copy a finish
                cudaStreamWaitEvent( stream_calculate , event_copy_matrix_a[ run_a ] , 0 );
            }
            // kernal add to stream_calculate for calculate data
            kernalMulMatrixTranspose<<< 1 , block_dimension , 0 , stream_calculate >>>(
                    d_matrix_a , d_matrix_b_t , d_matrix_c , column_a ,
                    run_a * grid_dimension.x , run_b * grid_dimension.y );
        }
        // Add event you can copy matrix c
        cudaEventRecord( event_copy_matrix_c[ run_a ] , stream_calculate );
        // stream copy c wait for can copy
        cudaStreamWaitEvent( stream_copy_c , event_copy_matrix_c[ run_a ] , 0 );
        // command add part copy matrix c to stream copy c
        // printf( "Add on position is %d\n" , run_a * size_copy_a );
        cudaMemcpyAsync( &ptr_matrix_c[ run_a * size_copy_a ],
                &d_matrix_c[ run_a * size_copy_a ],
                size_copy_c,
                cudaMemcpyDeviceToHost,
                stream_copy_c );
    }

    cudaEventRecord( stop , stream_copy_c );
    cudaDeviceSynchronize();
    float cpu_elapsed_time_ms;
    cudaEventElapsedTime(&cpu_elapsed_time_ms, start, stop);
    printf("Time elapsed on matrix multiplication of %dx%d . %dx%d on CPU: %f ms.\n\n", 
            row_a, column_a, 
            row_b, column_b,
            cpu_elapsed_time_ms );
    /*
    printf("Matrix_C\n");    
    printMatrix( ptr_matrix_c , row_a , column_b );
    */
    cudaFreeHost( ptr_matrix_a );
    cudaFreeHost( ptr_matrix_b );
    cudaFreeHost( ptr_matrix_c );
    cudaFreeHost( ptr_matrix_b_t );
    cudaFree( d_matrix_a );
    cudaFree( d_matrix_b_t );
    cudaFree( d_matrix_c );
    cudaStreamDestroy( stream_copy_a );
    cudaStreamDestroy( stream_copy_b );
    cudaStreamDestroy( stream_copy_c );
    cudaStreamDestroy( stream_calculate );
    return 0; 
}

void printMatrix( data_type* ptr_matrix , unsigned int row_size , unsigned int column_size )
{
    unsigned int index = 0;
    for( unsigned int row = 0 ; row < row_size ; row++ )
    {
        unsigned int limit = ( row + 1 ) * column_size;
        for( ; index < limit ; index++ )
        {
            printf( "%d " , ptr_matrix[ index ] );
        }
        printf("\n");
    }
}

void transposeMatrix( data_type* origin_matrix,
        data_type* transpose_matrix,
        unsigned int row_size,
        unsigned int column_size )
{
    for( unsigned int row = 0 ; row < row_size ; row++ )
    {
        unsigned int index = row * column_size;
        for( unsigned int column = 0 ; column < column_size ; column++ )
        {
            transpose_matrix[ column*row_size + row ] = origin_matrix[ index + column ];
        }
    }
}

void initMatrix( data_type* ptr_matrix, 
        unsigned int row_size, 
        unsigned int column_size, 
        data_type value )
{
    unsigned int limit_index = row_size * column_size;
    for( unsigned int index = 0 ; index < limit_index ; index++ )
    {
        ptr_matrix[ index ] = value;
    } 
}

void randomMatrix( data_type* ptr_matrix , unsigned int row_size , unsigned int column_size )
{
    unsigned int limit_index = row_size * column_size;
    for( unsigned int index = 0 ; index < limit_index ; index++ )
    {
        ptr_matrix[ index ] = rand() % 10;
    }
}
