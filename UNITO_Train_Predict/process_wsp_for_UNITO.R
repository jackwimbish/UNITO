# Install required packages if not already installed
if (!requireNamespace("BiocManager", quietly = TRUE)) {
  install.packages("BiocManager")
}
if (!requireNamespace("flowCore", quietly = TRUE)) {
  BiocManager::install("flowCore")
}
if (!requireNamespace("CytoML", quietly = TRUE)) {
  BiocManager::install("CytoML")
}
if (!requireNamespace("flowWorkspace", quietly = TRUE)) {
  BiocManager::install("flowWorkspace")
}

# Load libraries
library(flowCore)
library(CytoML)
library(dplyr)

# Function to process all FCS files in a folder with FlowJo workspace
process_fcs_folder_with_gates <- function(fcs_folder, workspace_file, output_folder, 
                                         filter_population = NULL, 
                                         selected_populations = NULL,
                                         selected_channels = NULL) {
  # Create output folder if it doesn't exist
  if (!dir.exists(output_folder)) {
    dir.create(output_folder, recursive = TRUE)
  }
  
  # Parse the FlowJo workspace
  ws <- open_flowjo_xml(workspace_file)
  
  # Get the sample groups in the workspace
  sample_groups <- fj_ws_get_sample_groups(ws)
  message("Available sample groups in workspace:")
  print(sample_groups)
  
  # Use the first group (index 1) as specified
  group_index <- 1
  sg_name <- sample_groups[group_index]
  message("Using sample group: ", sg_name)
  
  # Get the gating set for the group using the group index
  gs <- flowjo_to_gatingset(ws, name = group_index, path = fcs_folder)
  
  # Get the names of all populations (gates)
  all_pops <- gs_get_pop_paths(gs)
  message("Available populations (gates):")
  print(all_pops)
  
  # Check if filter_population exists
  if (!is.null(filter_population) && !(filter_population %in% all_pops)) {
    warning("Specified filter population '", filter_population, "' not found in workspace. Using all events.")
    filter_population <- NULL
  }
  
  # Add a message if filtering
  if (!is.null(filter_population)) {
    message("Filtering to include only events in population: ", filter_population)
  }
  
  # Determine which populations to include
  if (is.null(selected_populations)) {
    # Remove the first element which is usually the root node
    pop_paths <- all_pops[-1]
    message("Including all populations in output")
  } else {
    # Check if all requested populations exist
    missing_pops <- selected_populations[!(selected_populations %in% all_pops)]
    if (length(missing_pops) > 0) {
      warning("The following selected populations were not found in the workspace and will be skipped: ",
              paste(missing_pops, collapse = ", "))
    }
    
    # Use only populations that exist
    pop_paths <- selected_populations[selected_populations %in% all_pops]
    
    if (length(pop_paths) == 0) {
      stop("None of the selected populations were found in the workspace. Please check population names.")
    }
    
    message("Including ", length(pop_paths), " selected populations in output:")
    print(pop_paths)
  }
  
  # Get all sample names in the gating set
  sample_names <- sampleNames(gs)
  message("Found ", length(sample_names), " samples to process")
  
  # Process each sample in the gating set
  for (sample_name in sample_names) {
    message("Processing sample: ", sample_name)
    
    # Get the flowFrame for this sample
    # First get the cytoset for the root population
    cs <- gs_pop_get_data(gs[[sample_name]])
    
    # Convert cytoset to list and get the flowFrame
    cs_list <- cytoset_to_list(cs)
    ff <- cytoframe_to_flowFrame(cs_list[[1]])
    
    # Create an empty matrix to store gate memberships
    n_events <- nrow(ff)
    # If n_events is a list, take the first element
    if (is.list(n_events)) {
      n_events <- n_events[[1]]
    }
    n_pops <- length(pop_paths)
    gate_matrix <- matrix(0, nrow = n_events, ncol = n_pops)
    
    # Format population names for column headers
    formatted_pop_names <- sapply(pop_paths, function(pop) {
      # Replace slashes with underscores
      pop_name <- gsub("/", "_", pop)
      # Replace open parentheses with "p"
      pop_name <- gsub("\\(", "p", pop_name)
      # Replace close parentheses with "q"
      pop_name <- gsub("\\)", "q", pop_name)
      # Replace spaces with underscores
      pop_name <- gsub(" ", "_", pop_name)
      # Remove leading underscores
      pop_name <- gsub("^_+", "", pop_name)
      return(pop_name)
    })
    
    colnames(gate_matrix) <- formatted_pop_names
    
    # Fill the matrix with gate memberships
    for (i in 1:length(pop_paths)) {
      # Get indices of events in this population
      pop_indices <- gh_pop_get_indices(gs[[sample_name]], pop_paths[i])
      # Mark these events as members of this gate
      gate_matrix[pop_indices, i] <- 1
    }
    
    # Extract expression data
    expression_data <- exprs(ff)
    
    # Get marker names when available, otherwise use channel names
    channel_names <- colnames(expression_data)
    parameter_data <- parameters(ff)
    marker_names <- vector("character", length(channel_names))
    
    for (i in 1:length(channel_names)) {
      # Try to get marker name from the $desc field
      marker <- parameter_data$desc[i]
      # If desc is empty, try $name
      if (is.null(marker) || marker == "" || is.na(marker)) {
        marker <- parameter_data$name[i]
      }
      # If still empty, use the channel name
      if (is.null(marker) || marker == "" || is.na(marker)) {
        marker_names[i] <- channel_names[i]
      } else {
        marker_names[i] <- marker
      }
    }
    
    # Replace column names with marker names
    colnames(expression_data) <- marker_names
    
    # Filter channels if specified
    if (!is.null(selected_channels)) {
      # Check which channels exist
      available_channels <- colnames(expression_data)
      channel_name_lookup <- setNames(seq_along(available_channels), available_channels)
      
      # Try to match selected channels to both original channel names and marker names
      matched_indices <- c()
      for (ch in selected_channels) {
        # Exact match
        if (ch %in% available_channels) {
          matched_indices <- c(matched_indices, channel_name_lookup[ch])
        } else {
          # Partial match (check if any available channel contains this string)
          partial_matches <- grep(ch, available_channels, fixed = TRUE)
          if (length(partial_matches) > 0) {
            matched_indices <- c(matched_indices, partial_matches)
          }
        }
      }
      
      # Remove duplicates
      matched_indices <- unique(matched_indices)
      
      if (length(matched_indices) == 0) {
        warning("None of the selected channels were found in the data. Using all channels.")
      } else {
        message("Including ", length(matched_indices), " selected channels")
        expression_data <- expression_data[, matched_indices, drop = FALSE]
      }
    }
    
    # Combine expression data and gate memberships
    combined_data <- cbind(expression_data, gate_matrix)
    
    # Convert to data frame
    result_df <- as.data.frame(combined_data)
    
    # Apply population filter if specified
    if (!is.null(filter_population)) {
      # Get indices for the filter population
      filter_indices <- gh_pop_get_indices(gs[[sample_name]], filter_population)
      
      # Filter the dataframe to only include those indices
      result_df <- result_df[filter_indices, ]
      
      # Format the filter population for the filename
      filter_suffix <- filter_population
      # Replace slashes with underscores
      filter_suffix <- gsub("/", "_", filter_suffix)
      # Replace open parentheses with "p"
      filter_suffix <- gsub("\\(", "p", filter_suffix)
      # Replace close parentheses with "q"
      filter_suffix <- gsub("\\)", "q", filter_suffix)
      # Replace spaces with underscores
      filter_suffix <- gsub(" ", "_", filter_suffix)
      # Remove leading underscores
      filter_suffix <- gsub("^_+", "", filter_suffix)
      
      filter_suffix <- paste0("_", filter_suffix)
    } else {
      filter_suffix <- ""
    }
    
    # Clean up sample name (remove underscore and trailing digits, and remove ".fcs")
    clean_name <- gsub("_[0-9]+$", "", sample_name)
    clean_name <- gsub("\\.fcs$", "", clean_name)
    
    # Create output filename based on cleaned sample name and optional filter
    output_csv <- file.path(output_folder, 
                            paste0(clean_name, ".csv"))
    
    # Write to CSV
    write.csv(result_df, file = output_csv, row.names = FALSE)
    
    message("Created CSV: ", output_csv)
    message("  with ", nrow(result_df), " events and ", ncol(result_df), " columns")
  }
  
  message("All samples processed successfully")
}

# Wrapper function that automatically determines paths from workspace file
process_flowjo_workspace <- function(workspace_path, filter_population = NULL, 
                                    selected_populations = NULL,
                                    selected_channels = NULL) {
  # Get the directory containing the workspace file
  workspace_dir <- dirname(workspace_path)
  
  # Get the workspace filename without extension
  workspace_basename <- tools::file_path_sans_ext(basename(workspace_path))
  
  # Determine FCS folder path (same folder as workspace)
  fcs_folder <- file.path(workspace_dir, workspace_basename)
  
  # Determine output CSV folder path
  output_folder <- file.path(workspace_dir, paste0(workspace_basename, "_csvs"))
  
  message("Using workspace: ", workspace_path)
  message("FCS folder: ", fcs_folder)
  message("Output folder: ", output_folder)
  
  # Call the processing function with all parameters
  process_fcs_folder_with_gates(fcs_folder, workspace_path, output_folder, 
                               filter_population, selected_populations, selected_channels)
}

# Example usage
# Replace with your actual workspace path
#workspace_path <- "/path/to/01262024_B_Cell_Panel.wsp"

# Example 1: Process all events and all populations
# process_flowjo_workspace(workspace_path)

# Example 2: Filter to only include events in a specific population
# process_flowjo_workspace(workspace_path, 
#                         filter_population = "/Cells/Singlets/Lymphocytes/B Cells")

# Example 3: Select specific populations to include
# process_flowjo_workspace(workspace_path,
#                         selected_populations = c("/Cells/Singlets/Lymphocytes/B Cells",
#                                                "/Cells/Singlets/Lymphocytes/B Cells/IgD-CD27+"))

# Example 4: Select specific channels and populations
# process_flowjo_workspace(workspace_path,
#                         selected_populations = c("/Cells/Singlets/Lymphocytes/B Cells"),
#                         selected_channels = c("FSC-A", "SSC-A", "CD19", "CD27"))

# Example 5: Filter events and select specific populations and channels
# process_flowjo_workspace(workspace_path,
#                         filter_population = "/Cells/Singlets/Lymphocytes",
#                         selected_populations = c("/Cells/Singlets/Lymphocytes/B Cells",
#                                                "/Cells/Singlets/Lymphocytes/B Cells/IgD-CD27+"),
#                         selected_channels = c("FSC-A", "SSC-A", "CD19", "CD27", "IgD"))
