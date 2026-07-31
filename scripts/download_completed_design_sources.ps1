<#
.SYNOPSIS
Downloads the raw sources used by the completed pre-shock, tariff-weighted redesign.

.DESCRIPTION
Files are verified against the SHA-256 hashes recorded below. The download directory is
intentionally excluded from Git because the BACI archive is approximately 1.15 GB.
#>
[CmdletBinding()]
param(
    [string]$Destination
)

$ErrorActionPreference = 'Stop'

if ([string]::IsNullOrWhiteSpace($Destination)) {
    $Destination = Join-Path $PSScriptRoot '..\data\raw\capability_redesign_sources'
}
$Destination = [System.IO.Path]::GetFullPath($Destination)
New-Item -ItemType Directory -Force $Destination | Out-Null

$files = @(
    [pscustomobject]@{ Name = 'world_bank_historical_income_classifications.xlsx'; Uri = 'https://ddh-openapi.worldbank.org/resources/DR0095334/download'; Sha256 = 'f9e390f71327c9e0ae632a505a221157446c4c67bb79f7fae4645c6a392cc141' },
    [pscustomobject]@{ Name = 'BACI_HS12_V202601.zip'; Uri = 'https://www.cepii.fr/DATA_DOWNLOAD/baci/data/BACI_HS12_V202601.zip'; Sha256 = '2dd0dd5ae331a7ec08c8e3400b3ce7cb4201292f9626d3281bff65f82874e9b8' },
    [pscustomobject]@{ Name = 'ustr_2018_list1_notice.pdf'; Uri = 'https://ustr.gov/sites/default/files/2018-13248.pdf'; Sha256 = '3e27f0e9b7fc7ed8d26d017432f23d118d83e2c4d676bcbf08e8262170a6c67a' },
    [pscustomobject]@{ Name = 'ustr_2018_list2_notice.pdf'; Uri = 'https://ustr.gov/sites/default/files/enforcement/301Investigations/2018-17709.pdf'; Sha256 = '97a8baf56d3f6c96c4c0a0320265d403bf585a1d6a06dc124c1dfbf638281b61' },
    [pscustomobject]@{ Name = 'ustr_2018_list3_notice.pdf'; Uri = 'https://ustr.gov/sites/default/files/enforcement/301Investigations/83%20FR%2047974.pdf'; Sha256 = '1673b855361621f4ed324abcefbf18dfad2b7ee72b770a2d18399c57bb79f0eb' },
    [pscustomobject]@{ Name = 'ustr_2018_list3_modification.pdf'; Uri = 'https://ustr.gov/sites/default/files/enforcement/301Investigations/2018-21303.pdf'; Sha256 = '0df576cbba6164c1d0c6f9ec4c523b24448630ebcccd5a6b5b842ac4c4d377bc' },
    [pscustomobject]@{ Name = 'ustr_2019_list4_original_notice.pdf'; Uri = 'https://ustr.gov/sites/default/files/enforcement/301Investigations/Notice_of_Modification_%28List_4A_and_List_4B%29.pdf'; Sha256 = 'd0df1d776cb5706b3f4952dd44f5c31e30cc7d9dfa7e9f947e3ea7be7d7b791d' },
    [pscustomobject]@{ Name = 'ustr_2019_list4a_notice.pdf'; Uri = 'https://ustr.gov/sites/default/files/enforcement/301Investigations/Notice_of_Modification%E2%80%93August_2019.pdf'; Sha256 = 'bb10f597c99b0504d0a55a9c445199dc17b1e9d0d13161902c7e0394a16a7891' }
)

foreach ($file in $files) {
    $target = Join-Path $Destination $file.Name
    if (Test-Path -LiteralPath $target) {
        $existing = (Get-FileHash -Algorithm SHA256 -LiteralPath $target).Hash.ToLowerInvariant()
        if ($existing -eq $file.Sha256) {
            Write-Host "Verified existing $($file.Name)"
            continue
        }
        throw "Existing file hash does not match for $target. Remove it manually before rerunning."
    }

    Write-Host "Downloading $($file.Name)"
    Invoke-WebRequest -Uri $file.Uri -OutFile $target
    $actual = (Get-FileHash -Algorithm SHA256 -LiteralPath $target).Hash.ToLowerInvariant()
    if ($actual -ne $file.Sha256) {
        Remove-Item -LiteralPath $target -Force
        throw "SHA-256 verification failed for $($file.Name)."
    }
    Write-Host "Verified $($file.Name)"
}

Write-Host "All raw sources are available in $Destination"
