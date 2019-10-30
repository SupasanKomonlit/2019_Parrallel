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

typedef int data_type;

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
}

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
    unsigned int column_a = 64;
    unsigned int row_a = 64;
    unsigned int column_b = 64;
    unsigned int row_b = 64;
    unsigned int column_c = column_a;
    unsigned int row_c = row_b;
    
    data_type *ptr_matrix_a;
    data_type *ptr_matrix_b;
    data_type *ptr_matrix_b_t;
    data_type *ptr_matrix_c;

    ptr_matrix_a = ( data_type* ) malloc( column_a * row_a * sizeof( data_type ) );
    ptr_matrix_b = ( data_type* ) malloc( column_b * row_b * sizeof( data_type ) );    
    ptr_matrix_c = ( data_type* ) malloc( column_c * row_c * sizeof( data_type ) );
    ptr_matrix_b_t = ( data_type* ) malloc( column_b * row_b * sizeof( data_type ) );    

    randomMatrix( ptr_matrix_a, row_a , column_a );
    randomMatrix( ptr_matrix_b, row_b , column_b );
    initMatrix( ptr_matrix_c , row_c , column_c , 0 );
    transposeMatrix( ptr_matrix_b , ptr_matrix_b_t , row_b , column_b );

    printf("Matrix_a\n");    
    printMatrix( ptr_matrix_a , row_a , column_a );
    printf("Matrix_B\n");    
    printMatrix( ptr_matrix_b , row_b , column_b );

    mulMatrixTranspose( ptr_matrix_a , ptr_matrix_b_t , ptr_matrix_c ,
            row_a , column_a , column_b );

    printf("Matrix_C\n");    
    printMatrix( ptr_matrix_c , row_a , column_b );

    free( ptr_matrix_a );
    free( ptr_matrix_b );
    free( ptr_matrix_c );
    free( ptr_matrix_b_t );
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
