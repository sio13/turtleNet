from itertools import islice

def chunk(it, size: int):
    iter_list = iter(it)
    return iter(lambda: tuple(islice(iter_list, size)), ())

def save_collage(filepath: str,
                 batch: list,
                 rows: int,
                 columns: int,
                 width: int = 28,
                 height: int = 28,
                 interpolation: str = 'nearest',
                 cmap: str = 'grey'):
    batch = batch.reshape(batch.shape[0], width, height)
    fig, axs = plt.subplots(rows, columns)
    cnt = 0
    for i in range(rows):
        for j in range(columns):
            axs[i, j].imshow((batch[cnt] + 1) / 2., interpolation=interpolation, cmap=cmap)
            axs[i, j].axis('off')
            cnt += 1
    fig.savefig(f"{filepath}.png")
    plt.close()
