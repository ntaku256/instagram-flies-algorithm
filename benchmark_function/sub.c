#include <stdio.h>

int main() 
{
    int h;

    printf("勉強時間を入力\n");
    scanf("%d", &h);

    if (h >= 2) {
        printf("ゲームを1時間\n");
    }

    return 0;
}
