#include <bits/stdc++.h>

using namespace std;

constexpr int MAX_THREADS = (1 << 9);

struct edge {
    int u, v, w;
    edge(int u, int v, int w): u(u), v(v), w(w) {}
};

using comparison_func_t = bool (*) (edge*, edge*);

int n;
edge *edges, *chosen_edges;
int *par;
int num_edge;

__device__ bool comparison_weight(edge *x, edge *y) {
    if (x->w == y->w) {
        if (x->u == y->u)
            return x->v < y->v;
        return x->u < y->u;
    }
    return x->w < y->w;
}

__device__ bool comparison_node(edge *x, edge *y) {
    if (x->u == y->u)
        return x->v < y->v;
    return x->u < y->u;
}

__device__ comparison_func_t p_comparison_weight = comparison_weight;
__device__ comparison_func_t p_comparison_node = comparison_node;

int get_container_length(int x) {
    int ret = 1;
    while (ret < x)
        ret <<= 1;
    return ret;
}

__global__ void bitonic_sort_kernel(edge *d_edges, int j, int k, comparison_func_t comparison) {
    unsigned int i, ixj;
    i = threadIdx.x + blockDim.x * blockIdx.x;
    ixj = i ^ j;

    auto swap = [&](edge& x, edge& y)->void {
        edge temp = x;
        x = y;
        y = temp;
    };

    if (ixj > i) {
        if (((i & k) != 0) && (*comparison)(&d_edges[i], &d_edges[ixj]))
            swap(d_edges[i], d_edges[ixj]);
        else if (((i & k) == 0) && (*comparison)(&d_edges[ixj], &d_edges[i]))
            swap(d_edges[i], d_edges[ixj]);
    }
}

void bitonic_sort(edge *edges, int length, comparison_func_t comparison) {
    int container_length = get_container_length(length);
    for (int i = length; i < container_length; i++) {
        edges[i] = edge(INT_MAX, INT_MAX, INT_MAX);
    }
    length = container_length;

    edge *d_edges;
    size_t container_size = length * sizeof(edge);

    // Copy data to gpu
    cudaMalloc((void**) & d_edges, container_size);
    cudaMemcpy(d_edges, edges, container_size, cudaMemcpyHostToDevice);

    // Call kernel func
    int num_thread = min(length, MAX_THREADS);
    int num_blocks = length / num_thread;
    dim3 blocks(num_blocks, 1);
    dim3 threads(num_thread, 1);

    for (int k = 2; k <= length; k <<= 1) {
        for (int j = k >> 1; j > 0; j >>= 1) {
            bitonic_sort_kernel<<<blocks, threads>>>(d_edges, j, k, comparison);
        }
    }

    // Copy result from gpu
    cudaMemcpy(edges, d_edges, container_size, cudaMemcpyDeviceToHost);
    cudaFree(d_edges);
}

int main(int argc, char **argv) {
    // Copy function to device
    comparison_func_t h_comparison_weight;
    comparison_func_t h_comparison_node;

    cudaMemcpyFromSymbol(&h_comparison_weight, p_comparison_weight, sizeof(comparison_func_t));
    cudaMemcpyFromSymbol(&h_comparison_node, p_comparison_node, sizeof(comparison_func_t));

    // Init clock
    clock_t t = clock();

    // Input n
    cin >> n;

    // Initialize parents
    par = (int * ) malloc(n * sizeof(int));
    for (int i = 0; i < n; i++) {
        par[i] = i;
    }

    function<int(int)> find_set = [&](int x) {
        return (par[x] == x ? x : par[x] = find_set(par[x]));
    };

    function<bool(int, int)> merge_set = [&](int u, int v) {
        int pu = find_set(u), pv = find_set(v);
        if (pu == pv) return false;
        par[pv] = pu;
        return true;
    };

    // Input edge
    edges = (edge * ) malloc(n * n * sizeof(edge));
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            int x;
            cin >> x;
            if (x == -1 || i >= j) continue;
            edges[num_edge++] = edge(i, j, x);
        }
    }
    assert(num_edge >= n - 1);

    // Sort weight
    bitonic_sort(edges, num_edge, h_comparison_weight);

    // Build MST
    long long total_cost = 0;
    int num_chosen = 0;
    chosen_edges = (edge * ) malloc(num_edge * 2 * sizeof(edge));
    for (int i = 0; i < num_edge; i++) {
        int u = edges[i].u, v = edges[i].v, w = edges[i].w;
        if (merge_set(u, v)) {
            total_cost += w;
            chosen_edges[num_chosen++] = edges[i];
            if (num_chosen == n - 1) break;
        }
    }

    // Sort chosen edge for output
    bitonic_sort(chosen_edges, num_chosen, h_comparison_node);

    // Get duration
    double time_taken = ((double) (clock() - t)) / CLOCKS_PER_SEC;

    // Output
    cout << total_cost << '\n';
    for (int i = 0; i < num_chosen; i++) {
        cout << chosen_edges[i].u << '-' << chosen_edges[i].v << '\n';
    }
    cout << fixed << setprecision(12) << "Waktu eksekusi: " << time_taken << " ms\n";

    // Return
    return 0;
}
