#!/usr/bin/env bash

# NOTE: Using fMRIPrep 23.0.2 because AROMA was removed in later versions.

# List of subjects to process
subjects=("02" "03" "05" "06" "07")
nprocs=30
maxmem=200

# relevant directories
project_root="/LOCAL/ocontier/ffadims"
rawdatadir="${project_root}/data/fmri_experiment/bids"
derivsdir="${rawdatadir}/derivatives/fmriprep_aroma"
workdir_base="${project_root}/scripts/fmri_experiment/fmriprep_wdir_aroma_manual"
fs_license_dir="${project_root}/data/freesurfer_license"

# timing log file
timing_log="${project_root}/scripts/fmri_experiment/fmriprep_timing_log.txt"

# ensure derivatives directory exists
mkdir -p "${derivsdir}"
chmod -R a+w "${derivsdir}"

# Initialize timing log
echo "fMRIPrep Processing Times - $(date)" > "${timing_log}"
echo "==========================================" >> "${timing_log}"
echo "" >> "${timing_log}"

USER_ID="$(id -u)"
GROUP_ID="$(id -g)"

# Function to format duration in human readable format
format_duration() {
    local duration=$1
    local hours=$((duration / 3600))
    local minutes=$(((duration % 3600) / 60))
    local seconds=$((duration % 60))
    if [ $hours -gt 0 ]; then
        printf "%dh %dm %ds" $hours $minutes $seconds
    elif [ $minutes -gt 0 ]; then
        printf "%dm %ds" $minutes $seconds
    else
        printf "%ds" $seconds
    fi
}

# Function to create fresh working directory
create_workdir() {
    local subject="$1"
    local workdir="${workdir_base}"
    echo "Creating fresh working directory for subject ${subject}..." >&2
    # Remove existing working directory if it exists
    if [ -d "${workdir}" ]; then
        echo "Removing existing working directory: ${workdir}" >&2
        rm -rf "${workdir}"
    fi
    # Create new working directory and set permissions
    mkdir -p "${workdir}"
    chmod -R a+w "${workdir}"
    echo "${workdir}"
}

# Function to delete working directory
cleanup_workdir() {
    local workdir="$1"
    echo "Cleaning up working directory: ${workdir}"
    rm -rf "${workdir}"
}
# Record overall start time
overall_start=$(date +%s)
echo "Batch processing started at: $(date)" | tee -a "${timing_log}"
echo "" >> "${timing_log}"

# Loop through subjects
for subject in "${subjects[@]}"; do
    echo "========================================"
    echo "Processing subject: ${subject}"
    echo "========================================"
    
    # Record start time for this subject
    subject_start=$(date +%s)
    echo "Subject ${subject} started at: $(date)"
    
    # Create fresh working directory for this subject
    workdir=$(create_workdir "${subject}")
    
    echo "Starting fMRIPrep for subject '${subject}'..."
    
    # Run docker command directly
    docker run -ti --rm \
      --user ${USER_ID}:${GROUP_ID} \
      --memory="${maxmem}g" \
      -v "${rawdatadir}:/data:ro" \
      -v "${derivsdir}:/out" \
      -v "${workdir}:/work" \
      -v "${fs_license_dir}:/fs_license_dir" \
      nipreps/fmriprep:23.0.2 \
      /data /out participant \
      -w /work \
      --participant-label ${subject} \
      --fs-no-reconall \
      --use-aroma \
      --nprocs ${nprocs} \
      --output-spaces T1w func MNI152NLin2009cAsym \
      --fs-license-file /fs_license_dir/license.txt
    
    # Record end time and calculate duration
    subject_end=$(date +%s)
    subject_duration=$((subject_end - subject_start))
    formatted_duration=$(format_duration $subject_duration)
    
    # Check exit status
    exit_status=$?
    if [ $exit_status -eq 0 ]; then
        echo "fMRIPrep completed successfully for subject ${subject}"
        echo "Duration: ${formatted_duration}"
        echo "Subject ${subject}: SUCCESS - ${formatted_duration} ($(date))" >> "${timing_log}"
        # Clean up working directory on success
        cleanup_workdir "${workdir}"
        echo "Subject ${subject} processing complete."
    else
        echo "ERROR: fMRIPrep failed for subject ${subject} (exit code: ${exit_status})"
        echo "Duration before failure: ${formatted_duration}"
        echo "Subject ${subject}: FAILED (exit code: ${exit_status}) - ${formatted_duration} ($(date))" >> "${timing_log}"
        echo "Working directory preserved for debugging: ${workdir}"
        echo "Please check the logs and manually clean up when ready."
    fi
    
    echo ""
done

# Calculate and log overall duration
overall_end=$(date +%s)
overall_duration=$((overall_end - overall_start))
overall_formatted=$(format_duration $overall_duration)

echo "========================================"
echo "All subjects processed!"
echo "Total processing time: ${overall_formatted}"
echo "========================================"

# Final summary to log file
echo "" >> "${timing_log}"
echo "==========================================" >> "${timing_log}"
echo "Batch processing completed at: $(date)" >> "${timing_log}"
echo "Total processing time: ${overall_formatted}" >> "${timing_log}"
echo "==========================================" >> "${timing_log}"

echo "Timing log saved to: ${timing_log}"
