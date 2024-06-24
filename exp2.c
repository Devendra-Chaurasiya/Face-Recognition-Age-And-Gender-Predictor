#include <stdio.h>
#include <stdlib.h>
void quicksort(int *arr, int start, int end) {
    if (start >= end - 1) return;
    int pivot = arr[start], left = start + 1, right = end - 1;
    while (left < right) {
        while (arr[left] < pivot && left < end) left++;
        while (arr[right] > pivot && right > start) right--;
        if (left < right) {
            int temp = arr[left];
            arr[left++] = arr[right];
            arr[right--] = temp;
        }
    }
    int temp = arr[right];
    arr[right] = arr[start];
    arr[start] = temp;
    quicksort(arr, start, right);
    quicksort(arr, right + 1, end);
}

int main() {
    int n;
    printf("Enter the size of array: ");
    scanf("%d", &n);
    int *arr = malloc(sizeof(int) * n);
    for (int i = 0; i < n; i++) {
        printf("Enter %d Element: ", i + 1);
        scanf("%d", &arr[i]);
    }
    quicksort(arr, 0, n);
    printf("Sorted Array: ");
    for (int i = 0; i < n; i++) {
        printf("%d ", arr[i]);
    }
    printf("\n");
    free(arr);
    return 0;
}
