/*
 * main.c
 *
 *  Created on: Jan 16, 2013
 *      Author: hklingen
 *  modified Dec 13, 2014
 */
#include <stdlib.h>
#include <stdio.h>
#include <stdint.h>
#include <string.h>

void func_2d(char **p, size_t M, size_t N);

int main()
{
	char filename[] = "LSU_train.fasta";
	char **array; /*resulting 2d array*/
	char *buf;
	size_t bytes_read;
	int buffer_size = 81920; /*greater than longest header+sequence!*/
	int headstart = 0;
	int headend = 0;
	int seqcount = -1;
	int seqpos = 0;
	int lastfilepos = 0;
	int lastseq = 0;
	long long startPosition = 0;
	FILE *filestream;
	int nseq, nlength, cursor, i;


	/*************First pass to find seq length and num seq.********************************/
	printf("opening %s for first pass\n", filename);
	filestream = fopen(filename, "r");
	nseq = 0;
	nlength = 0;
	if (!filestream)
	{
		printf("cannot open %s\n", filename);
		return (1);
	}

	else
	{
		do {

			if (fseek(filestream, startPosition, SEEK_SET) == -1)
			{
				printf("fseek failed:\n");
				return (2);
			}
			buf = (char *)malloc(buffer_size * sizeof(char)); /*reserve memory for buffer*/
			bytes_read = fread(buf, 1, buffer_size, filestream); /*check if file is loaded and end is loaded as well*/

			for (cursor = 0; cursor < bytes_read; cursor++)
			{
				if (buf[cursor] == '>')
				{
					headstart = 1;
					seqcount++;
					if (seqpos > nlength)
					{
						nlength = seqpos;	/*set longest readlength*/
					}
					seqpos = 0;
				}

				else
					if ((buf[cursor] == '\n') && (headstart == 1))
					{
						headend = 1;
						headstart = 0;
					}

					else
						if (headend == 1 && headstart == 0)
						{
							if (buf[cursor] == '\n')
							{
								lastfilepos = cursor + 1;
								lastseq = seqcount;
							}
							seqpos++;
						}
			}
			startPosition = startPosition + (long long)lastfilepos; /*set position for next fseek*/
			seqcount = lastseq;
			nseq = seqcount;

			if (bytes_read != buffer_size) /*end of file is reached*/
			{
				startPosition = 0;
			}
			free(buf);
		} while (startPosition != 0);
		printf("closing file.\n");
		fclose (filestream);
	}

	nseq++; /*for malloc*/
	nlength++;
	printf("%d\n", nseq);
	array = malloc(nseq * sizeof(char *)); /*reserve memory for each sequence*/
	if (array == NULL)
	{
		fprintf(stderr, "out of memory\n");
		return (3);
	}
	int mali;
	for (mali = 0; mali < nseq; mali++)
	{
		array[mali] = malloc(nlength * sizeof(char));
		if (array[mali] == NULL)
		{
			fprintf(stderr, "out of memory\n");
			return (3);
		}
	}
	func_2d(array, nseq, nlength); /*Set array values to X for each Seq&Position*/

	seqcount = -1;
	seqpos = 0;

	/*************Second pass to fill 2d array.********************************/
	printf("opening %s for second pass\n", filename);
	filestream = fopen(filename, "r");
	if (!filestream)
	{
		printf("cannot open %s:\n", filename);
	}

	else
	{
		do {

			if (fseek(filestream, startPosition, SEEK_SET) == -1)
			{
				printf("fseek failed:\n");
			}
			buf = (char *)malloc(buffer_size * sizeof(char));
			bytes_read = fread(buf, 1, buffer_size, filestream);
			seqpos = 0;
			for (cursor = 0; cursor < bytes_read; cursor++)
			{
				if (buf[cursor] == '>')
				{
					headstart = 1;
					seqcount++;
					//printf("%d\n",seqcount);
					seqpos = 0;
				}


				else
					if ((buf[cursor] == '\n') && (headstart == 1))
					{
						headend = 1;
						headstart = 0;
					}

					else
						if (headend == 1 && headstart == 0)
						{
							if (buf[cursor] == '\n')
							{
								lastfilepos = cursor + 1;
								lastseq = seqcount;
								/*printf("%d\n",seqcount);*/
								seqpos--;
							}

							else
							{
								array[seqcount][seqpos] = buf[cursor];
							}
							seqpos++;
						}
			}

			startPosition = startPosition + (long long)lastfilepos;
			seqcount = lastseq;

			if (bytes_read != buffer_size)
			{
				startPosition = 0;
			}

		} while (startPosition != 0);
		printf("closing file.\t%d\n", seqcount);
		fclose (filestream);
	}
	for (i = 0; i < seqcount; i++)
	{
		free(array[i]);
	}

	free(array);
	printf("all done.\n");
	return (0);
}

void func_2d(char **p, size_t M, size_t N)
{
	size_t i, j;
	for (i = 0; i < M; i++)
	{
		for (j = 0; j < N; j++)
		{
			p[i][j] = 'X';/*set every position to 'X'*/
		}
		p[i][N] = '\0';
	}
}
